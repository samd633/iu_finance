# Methodology document generator

Builds the Word document that explains how the Quality + Growth models are
built, trained, tested and selected, filled with the results of a
`reports/combo_ic.py` run.

## One-time setup

```
cd reports/methodology_doc
npm install
```

Needs Node.js (for the `docx` package) and the project's Python environment.

## Use

From the project root, after running `python reports/combo_ic.py` (and, for the add-on section, `python reports/factor_ic.py` and `python reports/addon_ic.py`):

```
python reports/methodology_doc/make_doc.py
```

This writes `reports/<Sector>_Combo_Model_Methodology.docx`. Options:
`--notes FILE` (default `notes/<sector>.json`) and `--out FILE`.

## For another sector or industry group

1. In `reports/combo_ic.py`, set `SECTORS` (and `END_PERIOD`, windows or
   `CHOSEN` as needed) and run it.
2. Copy `notes/technology.json` to `notes/<sector>.json` and rewrite it for the
   new results. Every key is optional; a section is left out when its key is
   missing:
   - `prepared`: date shown on the title line
   - `coverage`: paragraphs on data coverage (Section 3.6)
   - `holdout_comment`: comment under the holdout table (Section 10.2)
   - `chosen_rationale`, `chosen_notes`: why the chosen models were picked
     (Section 11)
   - `addon_notes`: commentary on the value and momentum add-on test (Section 12)
   - `limitations`: sector-specific limitations (Section 13)
   - `dev_log`: `[change, reason]` rows for Appendix B
3. Run `make_doc.py`.

Everything else — sample statistics, settings, metric definitions, results
tables — comes from the code and the run's outputs.

## Files

- `make_doc.py`: gathers settings, sample statistics and results, then runs
  the builder.
- `build_doc.js`: lays out the document (`node build_doc.js <data.json>
  <notes.json> <out.docx>`).
- `notes/`: sector-specific prose, one file per sector.
