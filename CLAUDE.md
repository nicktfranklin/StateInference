# Claude Code Configuration

## Python Environment
- **IMPORTANT**: Always use `uv` for Python package management and code execution
- Use `uv run` to execute Python scripts instead of `python`
- Use `uv add` to install new dependencies instead of `pip install`
- Use `uv sync` to sync dependencies instead of `pip install -r requirements.txt`
- Python version: >=3.12 (as specified in pyproject.toml)

## Project Structure
- Package name: `valueiteration`
- Main source code: `src/value_iteration/`
- Models: `src/value_iteration/models/`
  - `mcts.py` - Monte Carlo Tree Search implementation
  - `sr.py` - Successor Representation models
  - `value_iteration_network.py` - Value Iteration Network models
- Utilities: `src/value_iteration/utils.py`
- Notebooks: `notebooks/` (Jupyter notebooks for experiments)

## Key Dependencies
- PyTorch (>=2.8.0) - Deep learning framework
- NumPy (>=2.3.2) - Numerical computing
- SciPy (>=1.16.1) - Scientific computing
- Matplotlib (>=3.10.6) - Plotting
- Seaborn (>=0.13.2) - Statistical visualization
- Jupyter (>=1.1.1) - Notebook environment
- tqdm (>=4.67.1) - Progress bars

## Available Models
- `MCTS`, `GridWorldNode`, `MctsSr` - Monte Carlo Tree Search
- `SRResampler` - Successor Representation
- `ValueIterationNetwork`, `UntiledValueIterationNetwork` - Value Iteration Networks

## Commands
- Run code: `uv run <script.py>`
- Run hello world test: `uv run hello.py`
- Install packages: `uv add <package_name>`
- Sync dependencies: `uv sync`
- Start Jupyter: `uv run jupyter notebook` or `uv run jupyter lab`

## Git Info
- Current branch: `main_sep_2025`
- Main branch for PRs: `master`