# Reports

Analyses of the prepared panel (`panel_prep.PanelPrep`). Scripts are run from
the project root; each writes its CSV outputs to a subfolder here. CSVs are
git-ignored, so rerun a script to regenerate them.

| Report | Script | Output folder | Published version |
|---|---|---|---|
| Tech Factor Signals: IC of each ranked metric vs the 1-year forward return rank (Technology, 2011Q4–2025Q2) | `python reports/factor_ic.py` | `technology_factor_ic_2025Q2/` | https://claude.ai/artifact/5RAP82dr97cVWoipP7TvmW (2026-10-11, commit a4f2fd5) |
| Tech Quality + Growth Models: IC of every 1–2 Quality x 1–2 Growth metric combination (Technology, 2011Q4–2025Q2); walk-forward size choice, nested rule, holdout 2011Q4–2020Q4 / 2021Q4–2025Q2, full-sample final picks and chosen models. Technology includes the companies moved in by `panel_prep.SECTOR_OVERRIDES` (Alphabet, Meta, Cerner, Xerox, Paychex, The Trade Desk and others; re-run 2026-10-28, chosen models unchanged). Methodology: `Technology_Combo_Model_Methodology.docx` (`python reports/methodology_doc/make_doc.py`) | `python reports/combo_ic.py` | `technology_combo_ic_2025Q2/` | https://claude.ai/artifact/67a471YTnZZcy9U7Qkon3t (2026-10-18, quarter-end returns from 2011Q4) |
| Tech Value + Momentum Add-ons: does adding a value or momentum metric (as configured or reversed, alone or one of each, full or half weight) improve the top-choice model? Paired IC gain, adjusted t, eras, holdout and walk-forward | `python reports/addon_ic.py` | `technology_addon_ic_2025Q2/` | https://claude.ai/artifact/67a471YTnZZcy9U7Qkon3t (add-on section, 2026-10-19) |
| Healthcare Factor Signals: IC of each ranked metric vs the 1-year forward return rank (Healthcare, 2011Q4–2025Q2) | `REPORT_SECTOR=Healthcare python reports/factor_ic.py` | `healthcare_factor_ic_2025Q2/` | in the Healthcare models dashboard below ("Every metric on its own") |
| Healthcare Quality + Growth Models: same test as Tech; chosen ROIC + FCF_Payout_Ratio_3Yr \| Asset_G_Rate_5Yr (top) and ROIC \| Asset_G_Rate_5Yr (second) | `REPORT_SECTOR=Healthcare python reports/combo_ic.py` | `healthcare_combo_ic_2025Q2/` | https://claude.ai/artifact/ABi2W7hTiz6DbtC2n3bT3A (2026-10-29) |
| Consumer Cyclical Factor Signals: IC of each ranked metric vs the 1-year forward return rank (Consumer Cyclical, 2011Q4–2025Q2; packaging makers moved to Basic Materials, DoorDash moved in) | `REPORT_SECTOR="Consumer Cyclical" python reports/factor_ic.py` | `consumer_cyclical_factor_ic_2025Q2/` | — |
| Consumer Cyclical Quality + Growth Models: same test as Tech, with payout yield and payout ratios as one family; chosen Payout_YLD \| EPS_Growth_1Yr + Inv_Rate (top) and Payout_YLD \| Inv_Rate (second); no value or momentum add-on helps either. Methodology: `Consumer_Cyclical_Combo_Model_Methodology.docx` | `REPORT_SECTOR="Consumer Cyclical" python reports/combo_ic.py` | `consumer_cyclical_combo_ic_2025Q2/` | — |
