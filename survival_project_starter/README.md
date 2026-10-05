# Pancreatic cancer survival analysis

Use Python 3.12 for this project. Create the environment and install the pinned
dependencies from the project directory:

```powershell
py -3.12 -m venv .venv-py312
.\.venv-py312\Scripts\python.exe -m pip install -r requirements.txt
```

Run the complete analysis and its tests:

```powershell
.\.venv-py312\Scripts\python.exe run_all.py
.\.venv-py312\Scripts\python.exe -m unittest discover -s tests -v
```

`run_all.py` reads the two cohort CSVs without modifying them and writes figures,
tables, and fold-level outputs to `results/`. The standalone
`survival_analysis.py` is a separate entry point that writes to `outputs/`; the
new workflow does not overwrite those outputs. Its Cox fits standardize
predictors using training-only statistics and increase regularization when
convergence needs it. Its coefficient estimates are per one standard deviation.
See `results_summary.md` for the results and interpretation.
