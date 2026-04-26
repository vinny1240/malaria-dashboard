# Malaria STELLA Dashboard

This dashboard reconstructs the STELLA malaria midterm model in Python and provides interactive visualizations for Tasks 1, 2.1, 2.2, 2.3, and 2.4.

## Files

- `app.py`: Streamlit dashboard UI
- `model_core.py`: RK4 ODE simulation core
- `requirements.txt`: Python dependencies

## Local setup

```bash
cd malaria_dashboard
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## Deployment with Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload `app.py`, `model_core.py`, and `requirements.txt` to the repo root.
3. Go to https://streamlit.io/cloud and connect your GitHub account.
4. Choose the repository and set the main file path to `app.py`.
5. Deploy. Streamlit will generate a public URL ending in `.streamlit.app`.

## Notes

- The app uses Python/RK4 simulation rather than STELLA native publishing.
- STELLA `.stmx` / `.isdb` files can be kept in the GitHub repo as source evidence, but they are not required for the dashboard to run.
- Default values match the midterm baseline and task settings.
