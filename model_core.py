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


def _safe_nonnegative(y: np.ndarray) -> np.ndarray:
    """Avoid tiny negative values from numerical integration."""
    return np.maximum(y, 0.0)


def derivatives(
    t: float,
    y: np.ndarray,
    p: Params,
    pesticide_p0: float = 0.0,
    pesticide_k: float = 1.4,
    warming_dc: float | None = None,
    base_temp: float = 22.0,
    c_max: float = 0.8,
    c_quad: float = 0.0125,
    z_max: float = 0.99,
    z_quad: float = 0.01,
) -> np.ndarray:
    """Core malaria ODEs matching the STELLA model logic."""
    H, S, M, I = _safe_nonnegative(y)
    N = H + S
    m = M + I

    c = p.c
    z = p.z
    if warming_dc is not None:
        current_temp = base_temp + warming_dc
        c = max(0.0, c_max - c_quad * (current_temp - 26.0) ** 2)
        z = max(0.0, z_max - z_quad * (current_temp - 25.0) ** 2)

    pesticide_effect = pesticide_p0 * np.exp(-pesticide_k * t)
    g_new = p.g + pesticide_effect

    # Human population flows
    infection_h_to_s = c * z * H * I
    recovery_s_to_h = p.r * S
    human_births = p.b * N
    human_natural_deaths_h = p.u * H
    human_natural_deaths_s = p.u * S
    malaria_deaths = p.i * S

    # Mosquito population flows
    mosquito_births = p.e * m
    mosquito_infection = (S / (N + 1.0)) * c * M
    mosquito_deaths_m = g_new * M
    mosquito_deaths_i = g_new * I

    dH = human_births + recovery_s_to_h - infection_h_to_s - human_natural_deaths_h
    dS = infection_h_to_s - recovery_s_to_h - malaria_deaths - human_natural_deaths_s
    dM = mosquito_births - mosquito_deaths_m - mosquito_infection
    dI = mosquito_infection - mosquito_deaths_i

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
    """Run RK4 simulation and return a tidy time-series DataFrame."""
    p = params or Params()
    n_steps = int(round(t_end / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    y = np.array([p.H0, p.S0, p.M0, p.I0], dtype=float)
    out = np.zeros((n_steps + 1, 4), dtype=float)
    out[0] = y

    for idx in range(n_steps):
        t = times[idx]
        k1 = derivatives(t, y, p, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        k2 = derivatives(t + dt / 2, y + dt * k1 / 2, p, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        k3 = derivatives(t + dt / 2, y + dt * k2 / 2, p, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        k4 = derivatives(t + dt, y + dt * k3, p, pesticide_p0, pesticide_k, warming_dc, base_temp, c_max, c_quad, z_max, z_quad)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = _safe_nonnegative(y)
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
    """Return common peak and final-state metrics."""
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
    """Run multiple simulations varying one Params field."""
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
