"""Run the original Brian2 model of Shiu et al. 2024 on the same experiment, for comparison.

Requires a clone of https://github.com/philshiu/Drosophila_brain_model (pass its path) and
`pip install brian2 joblib`. Slow on CPU (minutes per trial); meant as a one-off ground truth.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="path to the Drosophila_brain_model clone")
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--n-proc", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from brian2 import Hz, prefs  # noqa: E402
    from model import default_params, run_exp  # noqa: E402

    import utils as utl  # noqa: E402

    from flysurvivors.connectome import DATA_DIR  # noqa: E402
    from flysurvivors.neurons import MN9, SUGAR_GRN_RIGHT  # noqa: E402

    prefs.codegen.target = "numpy"
    comp = pd.read_csv(DATA_DIR / "Completeness_783.csv", index_col=0)
    sugar = [i for i in SUGAR_GRN_RIGHT if i in comp.index]

    params = dict(default_params)
    params["n_run"] = args.trials
    params["r_poi"] = args.rate * Hz
    out = Path("results/reference")
    name = f"sugarR_{int(args.rate)}Hz"
    run_exp(
        exp_name=name,
        neu_exc=sugar,
        path_res=str(out),
        path_comp=str(DATA_DIR / "Completeness_783.csv"),
        path_con=str(DATA_DIR / "Connectivity_783.parquet"),
        params=params,
        n_proc=args.n_proc,
        force_overwrite=True,
    )
    df = utl.load_exps([out / f"{name}.parquet"])
    rate, std = utl.get_rate(df, t_run=1.0, n_run=args.trials)
    rate = rate[name]
    print(f"MN9 rate: {rate.get(MN9, 0.0):.1f} Hz")
    non_stim = rate.drop(index=[i for i in sugar if i in rate.index])
    print(f"active non-stimulated neurons: {int((non_stim > 0).sum())}")
    print("top 15 non-stimulated neurons:")
    print(non_stim.sort_values(ascending=False).head(15).to_string())
    rate.to_csv(out / f"{name}_rates.csv")


if __name__ == "__main__":
    main()
