# Cog

An Intelligence Runtime with a scientific learning engine. Every task is an
experiment; every experiment produces evidence; every verified pattern
compresses into a better reasoning architecture.

This repository is **self-contained** so it can be cloned and run anywhere with
a single command — including Kaggle.

## Layout

```
cog/
├── cog/                       # the package (import as `import cog`)
│   ├── runtime/               # CogRuntime orchestration loop
│   ├── execution/             # planner, executor, ordering
│   ├── learning/              # policy/belief/calibration engines
│   ├── science/               # promotion + governance
│   └── ...
├── experiments/
│   └── exp_broad_validation.py   # broad ACTIVE-validation campaign (Kaggle target)
├── scripts/
│   ├── kaggle_setup.sh
│   └── kaggle_run.sh
├── tests/                     # (test discovery via pytest)
├── requirements.txt
├── run.py                     # single entry point
└── README.md
```

## Local usage

```bash
pip install -r requirements.txt

python run.py bench        # deterministic benchmark suite
python run.py broadval     # broad validation campaign (writes ./results.json)
python run.py broadval --n 100 --seeds 1 2 3
python run.py test         # run the test suite (requires pytest)
```

Or invoke the experiment directly:

```bash
python experiments/exp_broad_validation.py --n 300 --seeds 1 2 3
```

Results are written to `results.json` locally, or `/kaggle/working/results.json`
when running on Kaggle.

## Kaggle usage

In a Kaggle notebook:

```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
```

```bash
!bash scripts/kaggle_setup.sh
!bash scripts/kaggle_run.sh
```

`kaggle_run.sh` runs the broad-validation campaign and writes
`/kaggle/working/results.json`. When the notebook finishes, download everything
in `/kaggle/working`.

### One-click reruns

After the notebook is set up once, rerunning is just: open the notebook and
click **Run All**. Each run clones the latest code, installs dependencies, and
executes the benchmark automatically.

## Notes

- The `cog` package is **pure standard library** — no third-party runtime
  dependencies, so Kaggle install is near-instant.
- The broad-validation campaign is governed: it prints a GO/NO-GO for promoting
  the `policy_dep_aware_prioritization` heuristic to ACTIVE but does **not**
  auto-promote. Promotion requires explicit, provenance-recorded sign-off.
