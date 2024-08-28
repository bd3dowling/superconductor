# superconductor

Minimal superconductor design-bench experiment

To reproduce benchmark results, follow the instructions of this
[repo](https://github.com/kaist-silab/design-baselines-fixes).

To reproduce SMCDiffOpt results:

- Clone the repository:

```bash
git clone https://github.com/bd3dowling/superconductor.git
```

- Use [`poetry`](https://python-poetry.org/) to install dependencies into a virtual environment:

```bash
poetry install && poetry shell
```

- Run the scripts:

```bash
python superconductor.py
python superconductor-plot.py
```
