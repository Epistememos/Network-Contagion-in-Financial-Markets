# Network Contagion in Financial Markets

A Python research repository for exploring network contagion in equity markets using correlation graphs, minimum spanning trees, and temporal directed network methods.

## Project overview

This repository focuses on:
- building market networks from equity returns
- constructing minimum spanning trees (MSTs) to capture market structure
- tracing contagion paths and network centrality changes
- exploring temporal multiplex directed relationships between assets

## Repository structure

- `src/` — primary experiment script(s) and entry points
- `data/` — processing utilities, including MST support
- `py_scripts/project1/` — Project 1 research code for MST-based market contagion
- `py_scripts/project2/` — Project 2 research code for temporal directed financial networks
- `notebooks/` — interactive Jupyter notebooks for analysis and visualization
- `tests/` — test code and validation scripts

## Environment setup

Recommended Python version: `3.10` or later.

### 1. Create and activate a virtual environment

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Command Prompt:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 2. Install required dependencies

Install the repository dependencies from the provided requirements file:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional notebook support:

```powershell
python -m pip install notebook jupyterlab
```

### 3. Verify the environment

Run a quick import check:

```powershell
python -c "import numpy, pandas, networkx, yfinance, matplotlib; print('OK')"
```

## Running the code

### Run the main script

The repository includes `src/main.py`, which loads price data, computes returns, and builds MST-based analysis slices.

```powershell
python src\main.py
```

### Explore the notebooks

Open the notebooks for interactive analysis:

```powershell
jupyter notebook notebooks\MST.ipynb
```

or

```powershell
jupyter lab
```

## Notes and recommendations

- The code uses `yfinance` to download stock prices. Ensure you have internet access when running data downloads.
- If you prefer a locked dependency list, create `requirements.txt` after installing packages:

```powershell
python -m pip freeze > requirements.txt
```

- Use the virtual environment for all work in this repository to avoid conflicts with global packages.

## Next steps

- Add `requirements.txt` or `environment.yml` for reproducible setup
- Add entrypoint scripts or a `Makefile` for standard workflows
- Document dataset sources and expected input file formats if local data is used
