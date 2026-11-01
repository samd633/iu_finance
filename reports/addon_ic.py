"""
Add-on test: can a Value or Momentum signal improve a chosen model?

The base model (combo_ic.CHOSEN[BASE_ROLE]) is held fixed. Each add-on is a
third category added to its score with weight w relative to Quality and Growth
(1 = an equal third category, 0.5 = half that):

    Score = mean Quality bucket + mean Growth bucket + w x add-on bucket

Add-ons are every ranked Value and Momentum metric, as configured and reversed
("~"), alone or one Value with one Momentum metric (each at weight w).

The question for each add-on is whether it beats the base model, so the key
statistic is the paired Newey-West t of the quarterly IC difference
(add-on model minus base), Bonferroni-adjusted for the number of add-on models
tried. Results are reported on the full sample, by era, on the holdout split,
and in walk-forward testing: each fold picks the add-on with the best training
paired t (none unless it exceeds combo_ic.NESTED_T) and scores it on the next
year.

Run from the project root after combo_ic.py:
    python reports/addon_ic.py
for the top choice; set ADDON_BASE_ROLE to test another chosen model
(bash: ADDON_BASE_ROLE="Second choice" python ...).
Writes CSVs to reports/<sector>_addon_ic_<end period>/ for the top choice and
reports/<sector>_addon_ic_<end period>_<role>/ otherwise (git-ignored).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combo_ic import (CHOSEN, END_PERIOD, ERAS, GAP_QUARTERS, NESTED_T, SECTORS, TRAIN_END,  # noqa: E402
                      WF_FIRST_TRAIN_END, WF_STEP, bonferroni_t, nw_t_rows, spearman_rows)
from factor_ic import SECTOR_SLUG, nw_t  # noqa: E402
from metrics_cfg import load_metrics_cfg  # noqa: E402
from panel_prep import PanelPrep  # noqa: E402

BASE_ROLE = os.environ.get("ADDON_BASE_ROLE", "Top choice")
ADDON_GROUPS = ("Value", "Momentum")
WEIGHTS = (1.0, 0.5)
TOP_QUARTERLY = 5  # add-on models whose quarterly IC is written out


def main() -> None:
    prep = PanelPrep(sectors=SECTORS, end_period=END_PERIOD)
    d = prep.prepare()
    y_col = f"{prep.target.removesuffix('.Y')}.R.Y"
    cfg = load_metrics_cfg()["Metrics"]
    standardize = prep.standardize_map()
    n = prep.n_buckets

    def bucket(m: str) -> np.ndarray:
        """Rank bucket of m; "~m" flips actual ranks only (filled ranks keep their value)."""
        if m.startswith("~"):
            r = d[f"{m[1:]}.R"]
            return r.where(d[f"{m[1:]}.F"].eq(1), n + 1 - r).to_numpy(float)
        return d[f"{m}.R"].to_numpy(float)

    if BASE_ROLE not in CHOSEN:
        sys.exit(f"No {BASE_ROLE!r} model for {SECTORS[0]}; add it to combo_ic.CHOSEN_BY_SECTOR first.")
    base_combo = CHOSEN[BASE_ROLE]
    q, g = (part.split(" + ") for part in base_combo.split(" | "))
    base = np.mean([bucket(m) for m in q], axis=0) + np.mean([bucket(m) for m in g], axis=0)

    addons = {grp: [x for m in cfg[grp] if standardize.get(m) != "None" for x in (m, f"~{m}")] for grp in ADDON_GROUPS}
    sets = [(a,) for grp in ADDON_GROUPS for a in addons[grp]]
    sets += [(v, m) for v in addons["Value"] for m in addons["Momentum"]]
    models = [(s, w) for s in sets for w in WEIGHTS]
    cache = {m: bucket(m) for s in sets for m in s}
    scores = np.stack([base] + [base + w * sum(cache[m] for m in s) for s, w in models])

    period = d.index.get_level_values("Period")
    periods = sorted(period.unique())
    pidx = pd.PeriodIndex(periods)
    y = d[y_col].to_numpy(float)
    ic = np.empty((len(scores), len(periods)))
    for j, p in enumerate(periods):
        rows = np.flatnonzero(period == p)
        ic[:, j] = spearman_rows(scores[:, rows], y[rows])
    assert not np.isnan(ic).any(), "a quarter's IC is undefined"
    diff = ic[1:] - ic[0]

    train_end = pd.Period(TRAIN_END, "Q")
    train = np.asarray(pidx <= train_end)
    test = np.asarray(pidx > train_end + GAP_QUARTERS)
    out = pd.DataFrame({
        "addon": [" + ".join(s) for s, _ in models],
        "kind": ["Value + Momentum" if len(s) == 2 else ("Value" if s[0] in addons["Value"] else "Momentum")
                 for s, _ in models],
        "weight": [w for _, w in models],
        "full_ic": ic[1:].mean(axis=1),
        "full_t": nw_t_rows(ic[1:]),
        "gain": diff.mean(axis=1),
        "gain_t": nw_t_rows(diff),
        "gain_t_adj": bonferroni_t(nw_t_rows(diff), len(models)),
        "pct_quarters_better": (diff > 0).mean(axis=1),
        "train_gain": diff[:, train].mean(axis=1),
        "train_gain_t": nw_t_rows(diff[:, train]),
        "test_gain": diff[:, test].mean(axis=1),
        "test_gain_t": nw_t_rows(diff[:, test]),
        "test_ic": ic[1:, test].mean(axis=1),
        "worst_4q_ic": pd.DataFrame(ic[1:]).T.rolling(4).mean().min().to_numpy(),
    })
    era_start = pidx[0]
    for name, end in ERAS:
        era = np.asarray((pidx >= era_start) & (pidx <= pd.Period(end, "Q")))
        out[f"gain_{name}"] = diff[:, era].mean(axis=1)
        era_start = pd.Period(end, "Q") + 1
    out = out.sort_values("gain_t", ascending=False)

    base_row = {
        "model": base_combo, "full_ic": ic[0].mean(), "full_t": nw_t(pd.Series(ic[0])),
        "test_ic": ic[0, test].mean(), "worst_4q_ic": pd.Series(ic[0]).rolling(4).mean().min(),
        "addon_models": len(models),
    }
    era_start = pidx[0]
    for name, end in ERAS:
        era = np.asarray((pidx >= era_start) & (pidx <= pd.Period(end, "Q")))
        base_row[f"ic_{name}"] = ic[0, era].mean()
        era_start = pd.Period(end, "Q") + 1

    # Walk-forward: each fold keeps the add-on with the best training paired t, if above NESTED_T.
    folds, wf_rows = [], []
    t_end = pd.Period(WF_FIRST_TRAIN_END, "Q")
    while t_end + GAP_QUARTERS + 1 <= pidx[-1]:
        tr = np.asarray(pidx <= t_end)
        te = np.asarray((pidx > t_end + GAP_QUARTERS) & (pidx <= t_end + GAP_QUARTERS + WF_STEP))
        t = nw_t_rows(diff[:, tr])
        best = int(np.argmax(t))
        use = t[best] > NESTED_T
        pick = 1 + best if use else 0
        folds.append({"fold": len(folds) + 1, "train": f"{pidx[0]}-{t_end}", "test": f"{pidx[te][0]}-{pidx[te][-1]}",
                      "addon": out_label(models[best]) if use else "None (base model)",
                      "train_gain_t": t[best], "test_ic_base": ic[0, te].mean(), "test_ic_pick": ic[pick, te].mean()})
        wf_rows += [{"period": str(pidx[j]), "fold": len(folds), "ic_base": ic[0, j], "ic_pick": ic[pick, j]}
                    for j in np.flatnonzero(te)]
        t_end += WF_STEP
    folds, wf_q = pd.DataFrame(folds), pd.DataFrame(wf_rows)
    gain = wf_q["ic_pick"] - wf_q["ic_base"]
    wf_summary = pd.DataFrame([{
        "folds": len(folds), "folds_with_addon": int((folds["addon"] != "None (base model)").sum()),
        "wf_ic_base": wf_q["ic_base"].mean(), "wf_ic_pick": wf_q["ic_pick"].mean(),
        "wf_gain": gain.mean(), "wf_gain_t": nw_t(gain) if gain.abs().sum() else np.nan,
    }])

    top = out.head(TOP_QUARTERLY).index
    quarterly = pd.DataFrame({"period": [str(p) for p in periods], "base": ic[0]})
    for i in top:
        quarterly[out.at[i, "addon"] + f" (w={out.at[i, 'weight']:g})"] = ic[1 + i]

    role_suffix = "" if BASE_ROLE == "Top choice" else "_" + BASE_ROLE.lower().replace(" ", "_")
    dest = Path(__file__).resolve().parent / f"{SECTOR_SLUG}_addon_ic_{END_PERIOD}{role_suffix}"
    dest.mkdir(exist_ok=True)
    out.round(4).to_csv(dest / "addon_summary.csv", index=False)
    pd.DataFrame([base_row]).round(4).to_csv(dest / "addon_base.csv", index=False)
    folds.round(4).to_csv(dest / "addon_wf_folds.csv", index=False)
    wf_summary.round(4).to_csv(dest / "addon_wf_summary.csv", index=False)
    quarterly.round(4).to_csv(dest / "addon_quarterly.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 50)
    print(f"Base model ({BASE_ROLE}): {base_combo}")
    print(pd.DataFrame([base_row]).round(3).T.to_string(header=False))
    print(f"\nTop 20 of {len(models)} add-on models by paired t of the IC gain:")
    print(out.head(20).drop(columns=["train_gain", "test_gain_t"]).round(3).to_string(index=False))
    print("\nBest add-on by group:")
    print(out.groupby(["kind", "weight"]).head(1).drop(columns=["train_gain", "test_gain_t"]).round(3).to_string(index=False))
    print("\nWalk-forward:")
    print(folds.round(3).to_string(index=False))
    print(wf_summary.round(3).to_string(index=False))
    print(f"\nWrote {dest.relative_to(ROOT)}/")


def out_label(model) -> str:
    s, w = model
    return " + ".join(s) + f" (w={w:g})"


if __name__ == "__main__":
    main()
