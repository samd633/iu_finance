"""
Methodology Word document for a combo_ic.py run: how the models are built,
trained, tested and selected, with the run's results.

Reads the settings in reports/combo_ic.py (SECTORS, END_PERIOD, windows,
selection rules, CHOSEN), the run's CSV outputs, the prepared panel, and a
notes file of sector-specific prose (coverage caveats, why the chosen models
were picked, development log; see notes/technology.json). Then runs
build_doc.js (Node, `docx` package) to write the .docx.

Run combo_ic.py first, then from the project root:
    python reports/methodology_doc/make_doc.py [--notes FILE] [--out FILE]
Defaults: notes/<sector>.json and reports/<Sector>_Combo_Model_Methodology.docx.
One-time setup: `npm install` in reports/methodology_doc/.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))

import combo_ic  # noqa: E402
from factor_ic import NW_LAGS  # noqa: E402
from index_panel import START_PERIOD  # noqa: E402
from metrics_cfg import load_metrics_cfg  # noqa: E402
from panel_prep import EXCLUDED_TICKERS, INDUSTRY_GROUPS_FILE, PanelPrep  # noqa: E402


def quarter_end(p: pd.Period) -> str:
    d = p.end_time
    return f"{d.day} {d.strftime('%B %Y')}"


def groups_text(sectors: list[str]) -> str:
    cfg = load_metrics_cfg(str(ROOT / INDUSTRY_GROUPS_FILE))
    parts = []
    for sector in sectors:
        if sector not in cfg:
            parts.append(f"{sector} is a single industry group (not listed in industry_groups.JSON)")
        else:
            parts.append("; ".join(f"{g.replace('_', ' ')} ({', '.join(inds)})" for g, inds in cfg[sector].items()))
    return "Industry groups roll FMP industries up as defined in `industry_groups.JSON`: " + "; ".join(parts) + "."


def sample_stats(prep: PanelPrep, train_end: pd.Period, gap: int, eras: list) -> dict:
    d = prep.prepare()
    n = d.groupby(level="Period").size()
    test_start = train_end + gap + 1
    raw = prep.rank(prep.load())
    in_index = raw[raw["In_Index.D"].eq("Y")]
    excess = PanelPrep(sectors=prep.sectors, end_period=str(prep.end_period), target="Fwd_Excess_Return_1Yr.Y").prepare()
    both = d[[f"{prep.target.removesuffix('.Y')}.R.Y"]].join(excess[["Fwd_Excess_Return_1Yr.R.Y"]], how="inner")
    first_end, last_start = pd.Period(eras[0][1], "Q"), pd.Period(eras[-2][1], "Q") + 1
    y = d[prep.target]
    return dict(
        rows=len(d), companies=int(d.index.get_level_values("Ticker").nunique()), quarters=int(n.size),
        per_q_min=int(n.min()), per_q_med=float(n.median()), per_q_max=int(n.max()),
        train_rows=int(n[n.index <= train_end].sum()), train_med=float(n[n.index <= train_end].median()),
        gap_rows=int(n[(n.index > train_end) & (n.index < test_start)].sum()),
        test_rows=int(n[n.index >= test_start].sum()), test_med=float(n[n.index >= test_start].median()),
        groups={k: int(v) for k, v in d["Industry_Group.D"].value_counts().items()},
        y_buckets={str(k): int(v) for k, v in d[f"{prep.target.removesuffix('.Y')}.R.Y"].value_counts().sort_index().items()},
        y_median=float(y.median()), y_mean=float(y.mean()),
        inindex_rows=len(in_index), inindex_noY=int(in_index[prep.target].isna().sum()),
        offset_med=float(d["_offset_days.D"].abs().median()), offset_max=float(d["_offset_days.D"].abs().max()),
        excess_same=float((both.iloc[:, 0] == both.iloc[:, 1]).mean()),
        early_per_q=float(n[n.index <= first_end].median()), late_per_q=float(n[n.index >= last_start].median()),
    ), d


def main() -> None:
    sectors = combo_ic.SECTORS
    slug = "_".join(sectors)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", default=str(HERE / "notes" / f"{slug.lower()}.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / f"{slug}_Combo_Model_Methodology.docx"))
    args = ap.parse_args()

    out_dir = ROOT / "reports" / f"{slug.lower()}_combo_ic_{combo_ic.END_PERIOD}"
    if not (out_dir / "window.csv").exists():
        sys.exit(f"No combo_ic outputs in {out_dir}; run reports/combo_ic.py first.")
    notes_file = Path(args.notes)
    notes = json.loads(notes_file.read_text(encoding="utf-8")) if notes_file.exists() else {}
    if not notes:
        print(f"No notes file at {notes_file}; sector-specific sections will be left out.")

    end = pd.Period(combo_ic.END_PERIOD, "Q")
    train_end = pd.Period(combo_ic.TRAIN_END, "Q")
    gap = combo_ic.GAP_QUARTERS
    prep = PanelPrep(sectors=sectors, end_period=combo_ic.END_PERIOD)
    stats, d = sample_stats(prep, train_end, gap, combo_ic.ERAS)

    cfg = load_metrics_cfg()["Metrics"]
    singles = pd.read_csv(out_dir / "singles.csv").set_index("metric")
    metrics = [dict(cat=group, name=name, desc=m.get("Description", ""), formula=m["Formula"], std=m["Standardize"],
                    dir=m["Direction"], bottom=m.get("Bottom_If", ""), fill=float(d[f"{name}.F"].mean()),
                    **{k: float(singles.loc[name, k]) for k in ("train_ic", "train_t", "test_ic", "test_t")})
               for group in ("Quality", "Growth") for name, m in cfg[group].items()]
    reversed_ = [dict(name=m, **{k: float(singles.loc[m, k]) for k in ("train_ic", "train_t", "test_ic", "test_t")})
                 for m in singles.index if m.startswith("~")]
    audit = pd.read_csv(ROOT / "sp500_panel_audit.csv")["panel_status"].str.split(":").str[0].str.split(" (", regex=False).str[0]

    # Value and momentum (add-on candidates): definitions plus full-sample single-metric IC from factor_ic.py.
    factor_file = ROOT / "reports" / f"{slug.lower()}_factor_ic_{combo_ic.END_PERIOD}" / "ic_summary.csv"
    factor = pd.read_csv(factor_file).set_index("metric") if factor_file.exists() else pd.DataFrame()
    other = [dict(cat=group, name=name, desc=m.get("Description", ""), formula=m["Formula"], std=m["Standardize"],
                  dir=m["Direction"], fill=float(d[f"{name}.F"].mean()),
                  ic=float(factor.loc[name, "mean_ic"]) if name in factor.index else None,
                  t=float(factor.loc[name, "nw_t"]) if name in factor.index else None)
             for group in ("Value", "Momentum") for name, m in cfg.get(group, {}).items() if m["Standardize"] != "None"]

    # Add-on test (reports/addon_ic.py), if it has been run.
    addon_dir = ROOT / "reports" / f"{slug.lower()}_addon_ic_{combo_ic.END_PERIOD}"
    addons = None
    if (addon_dir / "addon_summary.csv").exists():
        addons = dict(base=pd.read_csv(addon_dir / "addon_base.csv").iloc[0].to_dict(),
                      top=pd.read_csv(addon_dir / "addon_summary.csv").head(10).replace({np.nan: None}).to_dict("records"),
                      wf=pd.read_csv(addon_dir / "addon_wf_summary.csv").iloc[0].to_dict())

    def rows(name):
        return pd.read_csv(out_dir / name).replace({np.nan: None}).to_dict("records")

    data = dict(
        stats=stats, metrics=metrics, reversed=reversed_, audit=audit.value_counts().to_dict(),
        other_metrics=other, addons=addons,
        results=dict(window=pd.read_csv(out_dir / "window.csv").iloc[0].to_dict(), sizes=rows("size_summary.csv"),
                     final=rows("final_models.csv"), chosen=rows("chosen.csv") if (out_dir / "chosen.csv").exists() else [],
                     candidates=rows("candidates.csv"), selection=rows("selection_summary.csv"), folds=rows("wf_folds.csv")),
        cfg=dict(
            sector=" + ".join(sectors), sectors=sectors, start=START_PERIOD, end=str(end),
            train_end=str(train_end), gap_quarters=gap, test_start=str(train_end + gap + 1),
            gap_window=f"{train_end + 1}-{train_end + gap}",
            ex_train_start=quarter_end(train_end + 1), ex_train_end=quarter_end(train_end + 5),
            last_price_date=quarter_end(end + 5),
            wf_first_train_end=combo_ic.WF_FIRST_TRAIN_END, wf_step=combo_ic.WF_STEP,
            nested_t=f"{combo_ic.NESTED_T:g}",
            nested_alt=["none" if a is None else f"{a:g}" for a in combo_ic.NESTED_T_ALTERNATIVES],
            max_per_cat=combo_ic.MAX_PER_CATEGORY, top_n=combo_ic.TOP_N, candidates=combo_ic.CANDIDATES,
            investment_family=list(combo_ic.INVESTMENT_FAMILY), nw_lags=NW_LAGS,
            eras=[name for name, _ in combo_ic.ERAS], excluded_tickers=list(EXCLUDED_TICKERS),
            groups_text=groups_text(sectors), out_dir=out_dir.relative_to(ROOT).as_posix(),
            today=date.today().strftime("%d %B %Y").lstrip("0"),
        ),
    )
    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        data_file, notes_used = Path(tmp) / "data.json", Path(tmp) / "notes.json"
        data_file.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        notes_used.write_text(json.dumps(notes, indent=1), encoding="utf-8")
        subprocess.run(["node", str(HERE / "build_doc.js"), str(data_file), str(notes_used), str(out)], check=True, cwd=HERE)
    register_font_table(out)
    print(f"Wrote {out}")


def register_font_table(path: Path) -> None:
    """docx (npm) writes word/fontTable.xml without a relationship to it, which
    strict validators flag; add the missing relationship."""
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        rels = zin.read("word/_rels/document.xml.rels").decode()
        if "fontTable" not in rels:
            rels = rels.replace("</Relationships>", '<Relationship Id="rIdFontTable1" Type="http://schemas.openxmlformats.org/'
                                "officeDocument/2006/relationships/fontTable\" Target=\"fontTable.xml\"/></Relationships>")
        for item in zin.infolist():
            zout.writestr(item, rels.encode() if item.filename == "word/_rels/document.xml.rels" else zin.read(item.filename))
    tmp.replace(path)


if __name__ == "__main__":
    main()
