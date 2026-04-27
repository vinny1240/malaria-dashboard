from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    H0: float = 100.0
    S0: float = 0.0
    M0: float = 1000.0
    I0: float = 10.0
    b: float = 0.05
    u: float = 0.02
    i: float = 0.5
    r: float = 0.25
    e: float = 0.202
    g: float = 0.2
    c: float = 0.6
    z: float = 0.9


def derivatives(
    t: float,
    y: np.ndarray,
    p: Params,
    dt: float,  # 新增 dt 參數以計算流量極限
    pesticide_p0: float = 0.0,
    pesticide_k: float = 1.4,
    warming_dc: float | None = None,
    base_temp: float = 22.0,
    c_max: float = 0.8,
    c_quad: float = 0.0125,
    z_max: float = 0.99,
    z_quad: float = 0.01,
) -> np.ndarray:
    """Core malaria ODEs implementing STELLA's strict non-negative flow restrictions."""
    H, S, M, I = np.maximum(y, 0.0) # 防止計算產生的微小負數
    N = H + S
    if N < 1e-12: 
        N = 1e-12
    m = M + I

    c = p.c
    z = p.z
    if warming_dc is not None:
        current_temp = base_temp + warming_dc
        c = max(0.0, c_max - c_quad * (current_temp - 26.0) ** 2)
        z = max(0.0, z_max - z_quad * (current_temp - 25.0) ** 2)

    pesticide_effect = pesticide_p0 * np.exp(-pesticide_k * t)
    g_new = p.g + pesticide_effect

    # 1. 初始計算所有「需求流量 (Requested Flows)」
    f_H_out_inf = c * z * H * I
    f_H_out_death = p.u * H
    
    f_S_out_rec = p.r * S
    f_S_out_death = (p.i + p.u) * S
    
    f_M_out_inf = (S / (N + 1.0)) * c * M
    f_M_out_death = g_new * M
    
    f_I_out_death = g_new * I

    # 2. 實作 STELLA 流量限制 (Flow Restriction)：確保流出量不會超過當下存量
    # H 的流量限制
    tot_H_out = f_H_out_inf + f_H_out_death
    if tot_H_out > H / dt and tot_H_out > 0:
        scale = (H / dt) / tot_H_out
        f_H_out_inf *= scale
        f_H_out_death *= scale
        
    # S 的流量限制
    tot_S_out = f_S_out_rec + f_S_out_death
    if tot_S_out > S / dt and tot_S_out > 0:
        scale = (S / dt) / tot_S_out
        f_S_out_rec *= scale
        f_S_out_death *= scale
        
    # M 的流量限制
    tot_M_out = f_M_out_inf + f_M_out_death
    if tot_M_out > M / dt and tot_M_out > 0:
        scale = (M / dt) / tot_M_out
        f_M_out_inf *= scale
        f_M_out_death *= scale
        
    # I 的流量限制
    if f_I_out_death > I / dt:
        f_I_out_death = I / dt

    # 3. 計算最終的淨變化率 (流入 - 流出)
    dH = p.b * N + f_S_out_rec - f_H_out_inf - f_H_out_death
    dS = f_H_out_inf - f_S_out_rec - f_S_out_death
    dM = p.e * m - f_M_out_inf - f_M_out_death
    dI = f_M_out_inf - f_I_out_death

    return np.array([dH, dS, dM, dI], dtype=float)


def simulate(
    params: Params | None = None,
    t_end: float = 100.0,
    dt: float = 1.0 / 64.0,
    pesticide_p0: float = 0.0,
    pesticide_k: float = 1.4,
    warming_dc: float | None = None,
    base_temp: float = 22.0,
    c_max: float = 0.8,
    c_quad: float = 0.0125,
    z_max: float = 0.99,
    z_quad: float = 0.01,
) -> pd.DataFrame:
    p = params or Params()
    n_steps = int(round(t_end / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    y = np.array([p.H0, p.S0, p.M0, p.I0], dtype=float)
    out = np.zeros((n_steps + 1, 4), dtype=float)
    out[0] = y

    for idx in range(n_steps):
        t = times[idx]
        # 傳遞 dt 進入 RK4 以進行流量限制
        k1 = derivatives(t, y, p, dt, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        k2 = derivatives(t + dt / 2, y + dt * k1 / 2, p, dt, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        k3 = derivatives(t + dt / 2, y + dt * k2 / 2, p, dt, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        k4 = derivatives(t + dt, y + dt * k3, p, dt, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)

        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.maximum(y, 0.0) # 最後的保險
        out[idx + 1] = y

    df = pd.DataFrame(out, columns=["H", "S", "M", "I"])
    df.insert(0, "time", times)

    if warming_dc is not None:
        current_temp = base_temp + warming_dc
        df["Current_Temp"] = current_temp
        df["c_T"] = max(0.0, c_max - c_quad * (current_temp - 26.0) ** 2)
        df["z_T"] = max(0.0, z_max - z_quad * (current_temp - 25.0) ** 2)

    if pesticide_p0 != 0:
        df["Pesticide_Effect"] = pesticide_p0 * np.exp(-pesticide_k * df["time"])
        df["g_new"] = p.g + df["Pesticide_Effect"]
    else:
        df["Pesticide_Effect"] = 0.0
        df["g_new"] = p.g

    return df


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    s_idx = int(df["S"].idxmax())
    i_idx = int(df["I"].idxmax())
    last = df.iloc[-1]

    return {
        "S_peak": float(df.loc[s_idx, "S"]),
        "S_time_to_peak": float(df.loc[s_idx, "time"]),
        "I_peak": float(df.loc[i_idx, "I"]),
        "I_time_to_peak": float(df.loc[i_idx, "time"]),
        "H_final": float(last["H"]),
        "S_final": float(last["S"]),
        "M_final": float(last["M"]),
        "I_final": float(last["I"]),
    }


def make_overlay(
    variable: str,
    values: Iterable[float],
    base_params: Params | None = None,
    t_end: float = 20.0,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p0 = base_params or Params()
    rows: List[pd.DataFrame] = []
    summary_rows = []

    for run_idx, val in enumerate(values, start=1):
        params_dict = p0.__dict__.copy()
        params_dict[variable] = float(val)
        p = Params(**params_dict)

        df = simulate(params=p, t_end=t_end, **kwargs)
        df["run"] = f"Run {run_idx}"
        df["varied_parameter"] = variable
        df["parameter_value"] = float(val)
        rows.append(df)

        summary = summarize(df)
        summary["run"] = f"Run {run_idx}"
        summary["varied_parameter"] = variable
        summary["parameter_value"] = float(val)
        summary_rows.append(summary)

    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary_rows)


def temperature_overlay(
    warming_values: Iterable[float],
    t_end: float = 40.0,
    base_params: Params | None = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []

    for run_idx, w in enumerate(warming_values, start=1):
        df = simulate(params=base_params or Params(), t_end=t_end, warming_dc=float(w), **kwargs)
        df["run"] = f"Run {run_idx}"
        df["Warming_dC"] = float(w)
        rows.append(df)

        summary = summarize(df)
        summary["run"] = f"Run {run_idx}"
        summary["Warming_dC"] = float(w)
        summary_rows.append(summary)

    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary_rows)
