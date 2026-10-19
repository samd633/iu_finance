# Reports

Analyses of the prepared panel (`panel_prep.PanelPrep`). Scripts are run from
the project root; each writes its CSV outputs to a subfolder here. CSVs are
git-ignored, so rerun a script to regenerate them.

| Report | Script | Output folder | Published version |
|---|---|---|---|
| Tech Factor Signals: IC of each ranked metric vs the 1-year forward return rank (Technology, 2011Q4–2025Q2) | `python reports/factor_ic.py` | `technology_factor_ic_2025Q2/` | https://claude.ai/artifact/5RAP82dr97cVWoipP7TvmW (2026-10-11, commit a4f2fd5) |
| Tech Quality + Growth Models: IC of every 1–2 Quality x 1–2 Growth metric combination (Technology, 2011Q4–2025Q2); walk-forward size choice, nested rule, holdout 2011Q4–2020Q4 / 2021Q4–2025Q2, full-sample final picks and chosen models. Methodology: `Technology_Combo_Model_Methodology.docx` (`python reports/methodology_doc/make_doc.py`) | `python reports/combo_ic.py` | `technology_combo_ic_2025Q2/` | https://claude.ai/artifact/67a471YTnZZcy9U7Qkon3t (2026-10-18, quarter-end returns from 2011Q4) |
| Tech Value + Momentum Add-ons: does adding a value or momentum metric (as configured or reversed, alone or one of each, full or half weight) improve the top-choice model? Paired IC gain, adjusted t, eras, holdout and walk-forward | `python reports/addon_ic.py` | `technology_addon_ic_2025Q2/` | https://claude.ai/artifact/67a471YTnZZcy9U7Qkon3t (add-on section, 2026-10-19) |
