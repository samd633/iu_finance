const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  HeadingLevel, AlignmentType, LevelFormat, PageOrientation, Footer, PageNumber, BorderStyle,
} = require("docx");

// Usage: node build_doc.js <docdata.json> <notes.json> <out.docx>  (make_doc.py runs this)
const D = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const NOTES = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const OUT = process.argv[4];

const FONT = "Calibri", MONO = "Consolas";
const f3 = v => (+v).toFixed(3), f2 = v => (+v).toFixed(2), pct = v => (100 * +v).toFixed(0) + "%", pct1 = v => (100 * +v).toFixed(1) + "%";
const int = v => Number(v).toLocaleString("en-US");

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold**, `code`, _italic_ inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: base.size ? base.size - 1 : 19, ...base, ...(base.font ? {} : {}) }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120 }, ...opts });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level }, spacing: { after: 60 } });
const N = (text, ref) => new Paragraph({ children: runs(text), numbering: { reference: ref, level: 0 }, spacing: { after: 60 } });
const EQ = text => new Paragraph({
  children: [new TextRun({ text, font: MONO, size: 19 })],
  indent: { left: 560 }, spacing: { before: 60, after: 120 },
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" },
});
const NOTE = text => new Paragraph({
  children: runs(text, { size: 19, color: "374151" }),
  indent: { left: 280 }, spacing: { before: 60, after: 160 },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: "6B7280", space: 8 } },
});

const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFC5CD" };
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const align = opts.align || header.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.LEFT));
  const size = opts.size || 18;
  const cell = (content, i, isHead, shade) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "1F3A5F", color: "auto" }
      : shade ? { type: ShadingType.CLEAR, fill: "F5F7FA", color: "auto" } : undefined,
    children: (Array.isArray(content) ? content : [content]).map(c =>
      c instanceof Paragraph ? c : new Paragraph({
        alignment: isHead ? AlignmentType.LEFT : align[i],
        children: isHead ? [new TextRun({ text: String(c), bold: true, color: "FFFFFF", size })] : runs(String(c), { size }),
      })),
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, k % 2 === 1)) })),
    ],
  });
}
const R = AlignmentType.RIGHT, L = AlignmentType.LEFT;
const gap = () => new Paragraph({ children: [], spacing: { after: 60 } });

// ---------- data ----------
const ST = D.stats, RS = D.results, W = RS.window, C = D.cfg, SEC = C.sector;
const Q = D.metrics.filter(m => m.cat === "Quality"), G = D.metrics.filter(m => m.cat === "Growth");
const groupMetrics = D.metrics.filter(m => m.std === "Industry_Group");
const maxFill = D.metrics.reduce((a, b) => (b.fill > a.fill ? b : a));
const sizeRows = rule => RS.sizes.filter(s => s.rule === rule);
const S0 = Object.fromEntries(sizeRows(W.rule).map(s => [s.n_metrics, s]));
const eraCols = C.eras.map(e => "ic_" + e);
const eraName = k => k.slice(3).replace("-", " to ");
const nQ = Q.length, nG = G.length, nRev = D.reversed.length, nGall = nG + nRev;
const qSets = nQ + nQ * (nQ - 1) / 2, gSets = nGall + nGall * (nGall - 1) / 2 - nRev;
const count2 = nQ * nGall, count3 = (nQ * (nQ - 1) / 2) * nGall + nQ * (nGall * (nGall - 1) / 2 - nRev), count4 = (nQ * (nQ - 1) / 2) * (nGall * (nGall - 1) / 2 - nRev);

const body = [];
const add = (...xs) => body.push(...xs);

// ---------- title ----------
add(
  new Paragraph({ children: [new TextRun({ text: `${SEC} Quality + Growth Models`, bold: true, size: 40, color: "1F3A5F" })], spacing: { after: 80 } }),
  new Paragraph({ children: [new TextRun({ text: "Model development, training and testing methodology", size: 28, color: "374151" })], spacing: { after: 120 } }),
  new Paragraph({
    children: [new TextRun({ text: `S&P 500 ${SEC} · ${W.sample.replace("-", " to ")} · Prepared ${NOTES.prepared || C.today} from reports/combo_ic.py`, size: 20, color: "6B7280" })],
    spacing: { after: 240 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3A5F", space: 6 } },
  }),
);

// ---------- 1 ----------
add(H1("1. Summary"));
add(P(`This document describes how the ${SEC} Quality + Growth models are built, trained, tested and selected, so the statistics and the development process can be reviewed in one place. Every number comes from the project code and its current outputs, and the code file responsible for each step is named throughout. Appendix B lists the design changes made during development and why.`));
add(P(`A **model** is a stock-ranking score that averages a company's quintile ranks on one or two Quality metrics and one or two Growth metrics. Each of the ${int(W.combinations)} possible models is scored by how well its ranking lines up with the ranking of the companies' next-year total returns, measured quarter by quarter with a Spearman information coefficient (IC). Every company's return is measured over the same calendar window, from the quarter-end after its financial data to one year later.`));
add(P("Model size (2, 3 or 4 metrics) is chosen by walk-forward testing: the selection rules are applied to an expanding training window and judged on the following year, nine times. The final models are then chosen on the full sample. A fixed holdout split is reported as an additional check."));
add(table(["Item", "Setting"], [
  ["Universe", `Point-in-time S&P 500 members (survivorship-free), ${SEC} (FMP classification)`],
  ["Sample", `${W.sample.replace("-", " to ")}: ${int(ST.rows)} company-quarters, ${ST.companies} companies, ${ST.quarters} quarters`],
  ["Dependent variable (Y)", "Quintile of 1-year total return from the calendar quarter-end after the period to one year later"],
  ["Predictors (X)", `Quintile ranks of ${nQ} Quality and ${nG} Growth metrics, plus ${nRev} Growth metrics also tried reversed`],
  ["Model score", "Mean Quality quintile + mean Growth quintile (each category weighted equally)"],
  ["Test statistic", "Per-quarter Spearman IC; mean IC; Newey-West t-stat (3 lags); Bonferroni-adjusted t"],
  ["Selection rules", `Rank by Newey-West t; one metric per family per category; nested rule (paired ${W.rule})`],
  ["Model size", `Chosen by walk-forward testing (${W.wf_folds} one-year test blocks, ${W.wf_test.replace("-", " to ")})`],
  ["Holdout check", `Train ${W.train.replace("-", " to ")} (${W.train_quarters} quarters), test ${W.test.replace("-", " to ")} (${W.test_quarters} quarters)`],
  ["Models tested", `${int(W.combinations)} (${int(count2)} with 2 metrics, ${int(count3)} with 3, ${int(count4)} with 4)`],
], [2600, 6760]));
add(gap());
if (RS.chosen.length) add(P("**Chosen models** (Section 11):"));
if (RS.chosen.length) add(table(["Role", "Quality", "Growth", "Full-sample IC", "t", "Adjusted t"],
  RS.chosen.map(c => [c.role, c.quality, c.growth, f3(c.full_ic), f2(c.full_t), f2(c.full_t_adj)]),
  [1500, 1900, 2900, 1200, 860, 1000], { align: [L, L, L, R, R, R] }));

// ---------- 2 ----------
add(H1("2. Pipeline overview"));
add(table(["Step", "Code", "What it does"], [
  ["1. Index membership", "sp_membership.py", "Point-in-time S&P 500 constituent history, including removed and delisted companies"],
  ["2. Raw data and metrics", "fmp_call.py, metrics.JSON", "Pulls statements and prices from FMP, builds trailing-twelve-month (TTM) fields, evaluates every metric and the forward-return target"],
  ["3. Panel", "index_panel.py → sp500_panel.csv", `Stacks every member company from ${C.start} into one (quarter, ticker) panel and writes an audit of inclusions and exclusions`],
  ["4. Preparation", "panel_prep.py", "Sector filter, industry groups, quintile ranks of X and Y, membership and target filters, missing-rank fill"],
  ["5. Single-metric test", "reports/factor_ic.py", "IC of each metric on its own (full sample)"],
  ["6. Model test and selection", "reports/combo_ic.py", "Scores every model, applies the selection rules, walk-forward and holdout tests, final and chosen models; writes all report CSVs"],
  ["7. Dashboard", "Claude artifact", "Visual summary of step 6"],
], [2100, 2700, 4560]));

// ---------- 3 ----------
add(H1("3. Data and universe"));
add(H2("3.1 Source"));
add(P("All company data come from the Financial Modeling Prep (FMP) API: quarterly income statements, balance sheets and cash-flow statements, company profiles (sector and industry), and daily closing prices, both split-adjusted and dividend-adjusted. API responses are cached locally for 24 hours (`fmp_cache.sqlite`)."));
add(H2("3.2 Survivorship-free universe"));
add(P(`The panel contains every company that was an S&P 500 member at any point since the fixed start period, ${C.start} (\`index_panel.START_PERIOD\`), including companies later removed, acquired or delisted. Membership history comes from the public fja05680/sp500 dataset, patched with Wikipedia's current constituent table, with ticker renames mapped through FMP's symbol-change history (\`sp_membership.py\`). Statements are fetched far enough back (about five extra years) for the 5-year metrics of the first quarter.`));
add(P("Each row carries `In_Index.D` = Y if the company was an index member on the **calendar quarter-end after the period**, the date its forward return starts (Section 4). Only those rows are used in testing."));
add(P(`Of the index symbols in the window, ${D.audit.included} are included, ${D.audit.failed} failed (FMP has no balance-sheet data for them), ${D.audit.excluded} are excluded (unresolved tickers or FMP data for a different company) and ${D.audit.dropped} are dropped (ticker reused, no membership in the window, or a second share class). Every outcome is recorded in \`sp500_panel_audit.csv\` and \`sp500_panel_exclusions.csv\`. Other exclusions:`));
add(B("Financial Services and Real Estate sectors: bank, insurer and REIT accounts don't fit the EBITDA, cash-flow, capex and leverage metrics."));
if (C.excluded_tickers.length) add(B(`Tickers whose FMP sector looks wrong (\`panel_prep.EXCLUDED_TICKERS\`): ${C.excluded_tickers.join(", ")}.`));
add(B("Companies with several share classes keep the one with the higher average daily dollar volume."));
add(H2("3.3 Calendarization (TTM)"));
add(P(`Each company's fiscal quarter-end is snapped to the nearest calendar quarter-end (dropped if more than 45 days away), so all companies in a period share the same economic quarter. In the sample the median snap is ${ST.offset_med} days and the largest ${ST.offset_max}. Flow items (income statement, cash flow) are trailing-twelve-month sums; balance-sheet items are the quarter-end snapshot. \`FIELD.N\` in a formula means N years back (4N quarterly rows).`));
add(H2("3.4 Point-in-time timing"));
add(P("A row's financial data are used only if its report was filed with the SEC on or before the calendar quarter-end that follows the period; otherwise the row is dropped. For example, financials for the quarter ending 31 March must be filed by 30 June. The forward return starts on that date, so every predictor was public when the return period began. Price-based metrics (value yields, momentum, payout yield) are measured on that same date: yields use the close at the start date, and momentum is the return over the 1, 3, 6, 12 or 24 months ending on it."));
add(H2(`3.5 The ${SEC} sample`));
add(table(["Measure", "Value"], [
  ["Company-quarter rows used", int(ST.rows)],
  ["Companies", String(ST.companies)],
  ["Quarters", `${ST.quarters} (${W.sample.replace("-", " to ")})`],
  ["Companies per quarter", `min ${ST.per_q_min}, median ${ST.per_q_med}, max ${ST.per_q_max} (median ${ST.early_per_q} in the first era, ${ST.late_per_q} in the last)`],
  ["Holdout training rows", `${int(ST.train_rows)} (median ${ST.train_med} companies per quarter)`],
  ["Holdout test rows", `${int(ST.test_rows)} (median ${ST.test_med} companies per quarter)`],
  ["Industry groups (rows)", Object.entries(ST.groups).map(([k, v]) => `${k.replace("_", " ")} ${int(v)}`).join(", ")],
], [3200, 6160]));
add(P(C.groups_text, { spacing: { before: 120, after: 120 } }));
if ((NOTES.coverage || []).length) { add(H2("3.6 Coverage")); NOTES.coverage.forEach(t => add(P(t))); }

// ---------- 4 ----------
add(H1("4. Dependent variable (Y)"));
add(H2("4.1 Forward 1-year total return on common quarter-end windows"));
add(P("For company i with financial data for calendar quarter t:"));
add(EQ("Fwd_Return_1Yr(i,t) = ( P_adj(i, QE(t+5)) / P_adj(i, QE(t+1)) − 1 ) × 100"));
add(P("QE(t+k) is the calendar quarter-end k quarters after quarter t, and P_adj is the **dividend-adjusted** close on the last trading day on or before that date, so the return includes dividends. Every company in a quarter is measured over exactly the same window: financials for the quarter ending 31 March 2020 pair with the return from 30 June 2020 to 30 June 2021. This matches a portfolio rebalanced once a quarter."));
add(P(`Y is missing (and the row dropped) if the report was filed after the start date, the company was not an index member on it, or there is no price within 7 days before either date (not yet listed, or delisted or acquired within the year). Of the ${int(ST.inindex_rows)} in-index rows up to 2025Q2, ${ST.inindex_noY} have no Y. The last usable quarter, ${C.end}, needs prices to ${C.last_price_date}.`));
add(H2("4.2 Quintile transformation"));
add(P(`Within each quarter, the in-index companies are sorted by return and split into five equal-count groups, 1 = highest return, using the same algorithm as the predictors (Section 5.2). Bucket counts are ${Object.values(ST.y_buckets).map(int).join(", ")}. Ranking within the quarter removes the market's overall level and limits the influence of extreme winners (raw median ${ST.y_median.toFixed(1)}%, mean ${ST.y_mean.toFixed(1)}%).`));
add(H2("4.3 Order of operations"));
add(P("Quintile breakpoints for Y and every predictor are computed first from in-index rows of the period; rows without Y or outside the index are dropped afterwards, so breakpoints don't depend on later survival. Companies acquired within the year therefore drop out of the IC (Section 13)."));
add(H2("4.4 Total vs excess return"));
add(P(`Because every company in a quarter shares one return window, subtracting the S&P 500 (SPY) return over that window subtracts the same number from every company and leaves the ranking unchanged. Checked on the data: ${(100 * ST.excess_same).toFixed(0)}% of rows fall in the same quintile under total and excess return. The choice therefore does not affect any IC in this report.`));

// ---------- 5 ----------
add(H1("5. Predictors (X)"));
add(H2("5.1 Candidate metrics"));
add(P(`${nQ} Quality metrics (returns on equity, assets and capital; margins; leverage; shareholder payout) and ${nG} Growth metrics (EPS, sales, cash-flow and payout growth; sustainable and internal growth; investment rate, capex intensity and asset growth) are candidates. Full definitions are in Appendix A.`));
add(P("Multi-year versions follow two conventions:"));
add(B("**Averages of a yearly ratio** for returns and margins (e.g. ROA_3Yr = mean of this year's and the previous two years' ROA)."));
add(B("**Ratio of multi-year totals** for payout ratios and capex intensity (e.g. CapX_Intensity_3Yr = 3-year capex / 3-year operating cash flow). A year with near-zero or negative earnings or cash flow would make its own ratio explode or go missing; dividing totals weights each year by its size."));
add(P("**Asset growth** follows Fama and French (2015): the growth in total assets, over one year or annualized over three and five years. It replaced an earlier PP&E-based definition that the 2019 lease-accounting change (ASC 842) distorted."));
add(P(`Investment-type metrics are configured "lower is better" (the conventional investment-factor direction). Because some predicted the other way in this sample, each of the ${nRev} is also tried **reversed** ("~" prefix): ${D.reversed.map(r => r.name.slice(1)).join(", ")}.`));
add(H2("5.2 How the quintiles are constructed"));
add(P("For metric m in quarter t and ranking group g (`PanelPrep.rank` and `_buckets`):"));
add(N("If the metric's direction is \"Lower\", negate the values so higher is always better.", "steps"));
add(N("The reference set is the non-missing values of in-index companies in (t, g), size N; ±infinity counts as missing.", "steps"));
add(N("Each company's average descending rank: r = (number of reference values greater than x) + (number equal to x + 1) / 2. Ties share a rank.", "steps"));
add(N("Bucket = ceiling( r × 5 / N − 10⁻⁹ ), limited to 1..5: 1 = best 20%, 5 = worst 20%. The 10⁻⁹ keeps a value exactly on a boundary in the better bucket.", "steps"));
add(N("Rows outside the index are bucketed against in-index breakpoints and dropped before testing.", "steps"));
add(P(`**Ranking group.** ${D.metrics.length - groupMetrics.length} metrics are ranked across all in-index companies in the sample each quarter; ${groupMetrics.length} (${groupMetrics.map(m => m.name).join(", ")}) are ranked within each of the four industry groups, because their typical levels differ structurally between industries.`, { spacing: { before: 120, after: 120 } }));
add(H2("5.3 Missing values"));
add(P("A missing rank is filled with bucket 5 (worst) if the metric's `Bottom_If` condition holds (e.g. `CFO_P <= 0`, currently burning operating cash, for cash-flow growth and capex intensity), and with bucket 3 (middle) otherwise. A flag column records every fill."));
add(P(`Fill rates range from 0% to ${pct1(maxFill.fill)} (${maxFill.name}); Appendix A lists each one.`));
add(H2("5.4 Reversed metrics"));
add(EQ("bucket(~m) = 6 − bucket(m)   for actual ranks;   filled ranks keep their filled value"));
add(P("Only actual ranks are flipped. A filled rank stays where the fill rule put it, so a cash-burning company placed in the worst bucket of capex intensity stays in the worst bucket of the reversed metric too. (An earlier version flipped filled ranks, which moved those companies to the best bucket; it was corrected.) A metric and its own reverse are never combined."));

// ---------- 6 ----------
add(H1("6. Model specification"));
add(P("Each model combines a set of Quality metrics Q (one or two) and a set of Growth metrics G (one or two). For company i in quarter t:"));
add(EQ("Score(i,t) = mean_{q in Q} bucket_q(i,t) + mean_{g in G} bucket_g(i,t)"));
add(P("Lower scores are better. The two categories carry equal weight whatever the number of metrics in each, and metrics within a category carry equal weight. The models have **no estimated coefficients**: training means choosing which metrics to combine."));
add(table(["Count", "Quality", "Growth", "Models"], [
  ["Candidate metrics", String(nQ), `${nGall} (${nG} + ${nRev} reversed)`, ""],
  ["Sets of one or two", int(qSets), `${int(gSets)} (pairs of a metric with its own reverse excluded)`, `${int(qSets)} × ${int(gSets)} = ${int(W.combinations)}`],
  ["2 metrics (1Q + 1G)", "", "", int(count2)],
  ["3 metrics (2Q + 1G or 1Q + 2G)", "", "", int(count3)],
  ["4 metrics (2Q + 2G)", "", "", int(count4)],
], [2900, 1500, 3060, 1900], { align: [L, R, R, R] }));

// ---------- 7 ----------
add(H1("7. Test statistics"));
add(H2("7.1 Information coefficient (IC), per quarter"));
add(EQ("IC(t) = Spearman correlation over companies i in quarter t of Score(i,t) and Y(i,t)"));
add(P("Spearman's correlation is the Pearson correlation of the ranks, with ties given their average rank. Both sides are ranked so that 1 is best, so a **positive IC means the best-scored companies tended to earn the best returns**. A mean IC of 0.05 to 0.10 sustained over time is usually considered useful for equity selection."));
add(H2("7.2 Mean IC and Newey-West t-statistic"));
add(P("The mean IC over a window weights each quarter equally. Consecutive quarters' 1-year returns overlap by nine months, so quarterly ICs are autocorrelated up to about three lags; the t-statistic uses a Newey-West (Bartlett) long-run variance with L = 3 (`factor_ic.nw_t`):"));
add(EQ("e(t) = IC(t) − mean IC          γ(l) = (1/T) Σ_t e(t)·e(t−l)"));
add(EQ("S = γ(0) + 2 Σ_{l=1..3} (1 − l/4)·γ(l)          t = mean IC / √(S / T)"));
add(H2("7.3 Paired t-statistic (nested rule)"));
add(P("To test whether a model improves on a smaller model it contains, the same Newey-West t is applied to the quarterly difference IC_model(t) − IC_smaller(t) over the selection window."));
add(H2("7.4 Bonferroni-adjusted t-statistic"));
add(P("Picking the best of many models inflates its t. The adjusted t converts the raw t to a one-sided p-value, multiplies it by the number of models M of that size that could have been picked, and converts back:"));
add(EQ("p = 1 − Φ(t)     p_adj = min(1, p × M)     t_adj = max( Φ⁻¹(1 − p_adj), 0 )"));
add(P(`M is the number of models of that size without a same-family pair: ${int(S0[2].models)} (2 metrics), ${int(S0[3].models)} (3) and ${int(S0[4].models)} (4). Bonferroni is conservative because the models are strongly correlated; an adjusted t above 2 is strong evidence.`));
add(H2("7.5 Other statistics"));
add(B("**Share of positive quarters**: a simple consistency measure."));
add(B(`**IC by era** (${C.eras.join(", ")}) and **worst four-quarter average IC**: stability across market regimes.`));
add(B("**Training-decile carry-over** and **selection comparison** on the holdout split."));
add(B("**Single-metric baselines** (Appendix A) and the single-metric report `factor_ic.py`."));

// ---------- 8 ----------
add(H1("8. Training and testing design"));
add(H2("8.1 Why a gap between training and test"));
add(P(`Each row's return spans four quarters. To keep test returns from sharing any days with training returns, ${C.gap_quarters} quarters are skipped between the last training quarter and the first test quarter. Example: the last holdout training quarter, ${C.train_end}, has its return from ${C.ex_train_start} to ${C.ex_train_end}; the first test quarter, ${C.test_start}, starts on ${C.ex_train_end}.`));
add(H2("8.2 Walk-forward testing (chooses model size)"));
add(P(`The training window starts at ${C.start} and expands a year at a time. In each fold the selection rules pick the best eligible model of each size (and the top 25) on the training window; their ICs are recorded on the next four quarters after the gap. The ${W.wf_folds} folds cover ${W.wf_test.replace("-", " to ")} out of sample (${W.wf_test_quarters} quarters), and each size's test ICs are strung together into one out-of-sample series.`));
add(table(["Fold", "Training window", "Test quarters"], RS.folds.map(f => [String(f.fold), f.train.replace("-", " to "), f.test.replace("-", " to ")]), [1200, 4080, 4080], { align: [L, L, L] }));
add(H2("8.3 Holdout split (additional check)"));
add(table(["Window", "Quarters", "Count", "Rows", "Used for"], [
  ["Training", W.train.replace("-", " to "), String(W.train_quarters), int(ST.train_rows), "Picking holdout models"],
  ["Gap", C.gap_window.replace("-", " to "), String(C.gap_quarters), int(ST.gap_rows), "Nothing"],
  ["Test", W.test.replace("-", " to "), String(W.test_quarters), int(ST.test_rows), "Out-of-sample check"],
], [1300, 2300, 900, 900, 3960], { align: [L, L, R, R, L] }));
add(H2("8.4 Why not a random split"));
add(P("A random split is not used. Companies in the same quarter share one return period, so splitting companies at random puts the same market outcome on both sides; splitting quarters at random puts overlapping 12-month returns on both sides. Either leaks information. Walk-forward testing with a gap gives several out-of-sample results without leakage."));
add(H2("8.5 Final selection on the full sample"));
add(P("Once walk-forward testing has shown that the selection method works and which size to use, the same rules are applied to all quarters to choose the final models. No untouched holdout remains for these, so their expected performance is judged from the walk-forward results, not from their full-sample statistics."));

// ---------- 9 ----------
add(H1("9. Selection rules"));
add(H2("9.1 Ranking"));
add(P("Models are ranked by Newey-West t-statistic on the selection window, which rewards a consistent IC rather than one driven by a few large quarters."));
add(H2("9.2 One metric per family"));
add(P("A model is excluded if one category contains two metrics of the same family, because they would double-count one signal:"));
add(B("horizons of one metric (ROE + ROE_3Yr);"));
add(B("earnings and cash versions of one ratio (FCF_MGN + CFO_MGN; CFOPS_CAGR + FCFPS_CAGR), including all six payout ratios (Payout_Ratio and FCF_Payout_Ratio at 1, 3 and 5 years);"));
add(B("**the investment family**: investment rate, capex intensity and total-asset growth at every horizon and in either direction. Pairing two of them in opposite directions (e.g. low investment rate with high asset growth) mostly hedged across market regimes rather than measuring anything coherent, so a model may hold at most one."));
add(H2("9.3 Nested rule (penalty for extra metrics)"));
add(P("AIC-type penalties don't apply because no parameters are estimated; the risk of overfitting comes from the number of models searched. A larger model is therefore only eligible if it beats **every** smaller model it contains (dropping one metric from a category with two) with a paired t above the threshold on the selection window."));
add(P(`The threshold in use is ${W.rule}. Walk-forward results under each threshold tested:`));
add(table(["Nested rule", "Metrics", "Eligible (full sample)", "Walk-forward IC", "t", "Folds won"],
  RS.sizes.map(s => [s.rule, String(s.n_metrics), int(s.full_eligible), s.wf_pick_ic === "" || s.wf_pick_ic === null ? "—" : f3(s.wf_pick_ic), s.wf_pick_t === "" || s.wf_pick_t === null ? "—" : f2(s.wf_pick_t), String(s.wf_folds_won)]),
  [1500, 1000, 2000, 1800, 1100, 1960], { align: [L, R, R, R, R, R] }));
{
  const none3 = RS.sizes.find(s => s.rule === "None" && s.n_metrics == 3);
  if (none3 && S0[3] && none3.wf_pick_ic != null && S0[3].wf_pick_ic != null && +none3.wf_pick_ic > +S0[3].wf_pick_ic)
    add(P(`Dropping the rule entirely gives a higher 3-metric walk-forward IC (${f3(none3.wf_pick_ic)} against ${f3(S0[3].wf_pick_ic)}), so the rule costs some out-of-sample performance here. It is kept as a guard against metrics that add nothing to a model; this trade-off is worth revisiting as more data accumulate.`, { spacing: { before: 120, after: 120 } }));
}
add(H2("9.4 Model size"));
{
  const rec = sizeRows(W.rule).find(s => s.recommended === "Yes");
  const others = sizeRows(W.rule).filter(s => s !== rec && s.wf_pick_ic != null).map(s => `${s.n_metrics} metrics (${f3(s.wf_pick_ic)})`);
  add(P(rec ? `Size is chosen by the walk-forward IC of each size's pick under the nested rule in use (${W.rule}). ${rec.n_metrics}-metric models do best (IC ${f3(rec.wf_pick_ic)}, t ${f2(rec.wf_pick_t)})${others.length ? ", ahead of " + others.join(" and ") : ""}.` : "No size was picked in every walk-forward fold."));
}

// ---------- 10 ----------
add(H1("10. Results"));
add(H2(`10.1 Final models (full sample, nested rule ${W.rule})`));
add(table(["ID", "Quality", "Growth", "IC", "t", "Adj. t", "Qtrs > 0"],
  RS.final.map(c => [c.candidate, c.quality, c.growth, f3(c.full_ic), f2(c.full_t), f2(c.full_t_adj), pct(c.full_pct_pos)]),
  [600, 2500, 3060, 800, 760, 800, 840], { align: [L, L, L, R, R, R, R], size: 16 }));
add(gap());
add(H2(`10.2 Holdout models (picked on ${W.train.replace("-", " to ")})`));
add(table(["ID", "Model", "Train IC", "Train t", "Test IC", "Test t", "Test qtrs > 0"],
  RS.candidates.map(c => [c.candidate, c.combo, f3(c.train_ic), f2(c.train_t), f3(c.test_ic), f2(c.test_t), pct(c.test_pct_pos)]),
  [600, 4300, 860, 800, 860, 800, 1140], { align: [L, L, R, R, R, R, R], size: 16 }));
if (NOTES.holdout_comment) add(P(NOTES.holdout_comment, { spacing: { before: 120, after: 120 } }));
add(H2("10.3 What selection buys (holdout split)"));
add(table(["Group", "Median test IC"], RS.selection.map(s => [s.group, f3(s.test_ic_median)]), [6400, 2960], { align: [L, R] }));
add(gap());

// ---------- 11 ----------
add(H1("11. Chosen models"));
if (!RS.chosen.length) add(P("No models have been chosen yet (CHOSEN in combo_ic.py is empty)."));
else {
  const row = (label, f) => [label, ...RS.chosen.map(f)];
  const w = Math.floor(6000 / RS.chosen.length);
  add(table(["", ...RS.chosen.map(c => c.role)], [
    row("Quality", c => c.quality),
    row("Growth", c => c.growth),
    row("Rank among eligible models of its size", c => c.full_rank_size == null ? "—" : String(c.full_rank_size)),
    row("Full-sample IC / t / adjusted t", c => `${f3(c.full_ic)} / ${f2(c.full_t)} / ${f2(c.full_t_adj)}`),
    row("Quarters with positive IC", c => pct(c.full_pct_pos)),
    ...eraCols.map(k => row(`IC ${eraName(k)}`, c => f3(c[k]))),
    row("Worst four-quarter average IC", c => f3(c.worst_4q_ic)),
    row("Paired t vs its smaller models (worst)", c => c.full_sub_t == null ? "—" : f2(c.full_sub_t)),
    row("Holdout: train IC / t", c => `${f3(c.train_ic)} / ${f2(c.train_t)}`),
    row("Holdout: test IC / t / quarters > 0", c => `${f3(c.test_ic)} / ${f2(c.test_t)} / ${pct(c.test_pct_pos)}`),
  ], [9360 - w * RS.chosen.length, ...RS.chosen.map(() => w)], { align: [L, ...RS.chosen.map(() => L)] }));
  if ((NOTES.chosen_rationale || []).length) { add(H2("11.1 Why these models")); NOTES.chosen_rationale.forEach(t => add(B(t))); }
  (NOTES.chosen_notes || []).forEach(t => add(P(t)));
}

// ---------- 12 ----------
// ---------- 12 ----------
const AD = D.addons;
add(H1("12. Value and momentum add-ons"));
if (!AD) add(P("No add-on test has been run (reports/addon_ic.py)."));
else {
  add(P(`The top-choice model is held fixed and each Value or Momentum metric is added as an extra category, as configured or reversed ("~"), alone or one of each, at full or half the weight of the Quality and Growth categories (\`reports/addon_ic.py\`). That is ${int(AD.base.addon_models)} extended models. For each, the quarterly IC gain over the base model is tested with the paired Newey-West t, adjusted (Bonferroni) for the number of add-ons tried, by era, on the holdout split, and in walk-forward testing: each fold keeps the add-on with the best training paired t if it exceeds the nested-rule threshold.`));
  add(EQ("Score = mean Quality + mean Growth + w × add-on bucket(s)   (w = 1 or 0.5)"));
  add(P(`Base model: ${AD.base.model} (IC ${f3(AD.base.full_ic)}, t ${f2(AD.base.full_t)}).`));
  const eraKeys = C.eras.map(e => "gain_" + e);
  add(table(["Add-on", "w", "IC gain", "Paired t", "Adj. t", ...C.eras.map(e => `Gain ${e.slice(0, 4)}–${e.slice(9, 13)}`), "Holdout test gain"],
    AD.top.map(a => [a.addon, String(a.weight), f3(a.gain), f2(a.gain_t), f2(a.gain_t_adj), ...eraKeys.map(k => f3(a[k])), f3(a.test_gain)]),
    [2800, 500, 900, 900, 800, ...C.eras.map(() => 900), 760].map((w, i, arr) => i === 0 ? 9360 - arr.slice(1).reduce((x, y) => x + y, 0) : w),
    { align: [L, R, R, R, R, ...C.eras.map(() => R), R], size: 16 }));
  add(P(`Walk-forward: the fold-picked add-on gave an out-of-sample IC of ${f3(AD.wf.wf_ic_pick)} against ${f3(AD.wf.wf_ic_base)} for the base model (gain ${f3(AD.wf.wf_gain)}, t ${f2(AD.wf.wf_gain_t)}); an add-on was used in ${AD.wf.folds_with_addon} of ${AD.wf.folds} folds.`, { spacing: { before: 120, after: 120 } }));
  (NOTES.addon_notes || []).forEach(t => add(P(t)));
}

add(H1("13. Limitations and review points"));
add(B(`**Multiple testing.** ${int(W.combinations)} models are searched. Raw t-stats are inflated; rely on the adjusted t and the walk-forward results.`));
add(B(`**Short, overlapping samples.** ${ST.quarters} quarters of overlapping one-year returns are about ${Math.round(ST.quarters / 4)} independent years; walk-forward covers about ${Math.round(W.wf_test_quarters / 4)}.`));
(NOTES.limitations || []).forEach(t => add(B(t)));
add(B("**Companies without a forward return.** Companies acquired within the year are dropped after ranking; if acquisition targets differ systematically, the IC is biased."));
add(B("**Classification.** FMP's current sector and industry are applied to all past periods."));
add(B("**Coarse quintiles and filled ranks.** Both sides are quintiles; missing values become bucket 3 or 5."));
if (RS.chosen.length) add(B("**Selection after review.** The chosen models were picked after seeing the full sample, including the walk-forward results. Their full-sample statistics overstate what to expect."));
add(B("**IC is not portfolio return.** It ignores position sizing, turnover and costs."));

// ---------- 13 ----------
add(H1("14. Reproducing the results"));
add(P("From the project root:"));
add(EQ("python index_panel.py          # rebuild the panel (slow)"));
add(EQ("python reports/factor_ic.py    # single-metric IC report"));
add(EQ("python reports/combo_ic.py     # model test, selection and all dashboard CSVs"));
add(EQ("python reports/addon_ic.py     # value and momentum add-ons to the top choice"));
add(EQ("python reports/methodology_doc/make_doc.py   # this document"));
add(table(["Parameter", "Value", "Meaning"], [
  ["START_PERIOD (index_panel.py)", C.start, "First quarter of the panel"],
  ["SECTORS", JSON.stringify(C.sectors), "Sectors kept before ranking"],
  ["END_PERIOD", C.end, "Last quarter used"],
  ["n_buckets", "5", "Quintiles"],
  ["target", "Fwd_Return_1Yr.Y", "Quarter-end 1-year total return"],
  ["TRAIN_END / GAP_QUARTERS", `${C.train_end} / ${C.gap_quarters}`, "Holdout split and gap"],
  ["WF_FIRST_TRAIN_END / WF_STEP", `${C.wf_first_train_end} / ${C.wf_step}`, "Walk-forward start and block length"],
  ["NESTED_T (alternatives)", `${C.nested_t} (${C.nested_alt.join(", ")})`, "Nested-rule threshold"],
  ["MAX_PER_CATEGORY", String(C.max_per_cat), "Most metrics per category"],
  ["INVESTMENT_FAMILY", C.investment_family.join(", "), "One investment metric per model"],
  ["TOP_N / CANDIDATES", `${C.top_n} / ${C.candidates}`, "Comparison group size; picks per size"],
  ["CHOSEN", RS.chosen.length ? RS.chosen.map(c => c.role).join(", ") : "none", "Reported in chosen.csv"],
  ["NW_LAGS (factor_ic.py)", String(C.nw_lags), "Newey-West lags"],
], [3000, 2600, 3760]));
add(gap());
add(P(`Outputs go to \`${C.out_dir}/\`: combos.csv (all models), final_models.csv, chosen.csv, candidates.csv, size_summary.csv, wf_folds.csv, wf_picks.csv, wf_quarterly.csv, top_combos.csv, selection_summary.csv, deciles.csv, singles.csv, top1pct_metrics.csv, window.csv and the quarterly IC files. CSVs are git-ignored; rerun to regenerate them.`));

// ---------- Appendix (landscape) ----------
const app = [];
app.push(H1("Appendix A. Metric definitions"));
app.push(P("Formulas are evaluated on TTM fields; `.N` means N years back. Group: rank within the whole sample (Universe) or within industry group. Better: which end ranks as bucket 1. Missing → 5 if: condition under which a missing rank becomes bucket 5 (otherwise 3). Filled: share of sample rows with a filled rank. IC columns are single-metric results on the holdout training and test windows."));
const mrow = m => [
  [new Paragraph({ children: [new TextRun({ text: m.name, bold: true, size: 16 })] })],
  [new Paragraph({ children: [new TextRun({ text: m.desc, size: 16 })] }),
   new Paragraph({ children: [new TextRun({ text: m.formula, font: MONO, size: 14, color: "4B5563" })] })],
  m.std === "Industry_Group" ? "Industry group" : "Universe", m.dir, m.bottom ? "`" + m.bottom + "`" : "", pct1(m.fill),
  f3(m.train_ic), f3(m.test_ic),
];
const widthsA = [1900, 6150, 1250, 800, 1300, 760, 760, 760];
const alignA = [L, L, L, L, L, R, R, R];
app.push(H2("Quality metrics"));
app.push(table(["Metric", "Definition and formula", "Group", "Better", "Missing → 5 if", "Filled", "Train IC", "Test IC"], Q.map(mrow), widthsA, { align: alignA, size: 16 }));
app.push(H2("Growth metrics"));
app.push(table(["Metric", "Definition and formula", "Group", "Better", "Missing → 5 if", "Filled", "Train IC", "Test IC"], G.map(mrow), widthsA, { align: alignA, size: 16 }));
const orow = m => [
  [new Paragraph({ children: [new TextRun({ text: m.name, bold: true, size: 16 })] })],
  [new Paragraph({ children: [new TextRun({ text: m.desc, size: 16 })] }),
   new Paragraph({ children: [new TextRun({ text: m.formula, font: MONO, size: 14, color: "4B5563" })] })],
  m.std === "Industry_Group" ? "Industry group" : "Universe", m.dir, pct1(m.fill), m.ic == null ? "—" : f3(m.ic), m.t == null ? "—" : f2(m.t),
];
(D.other_metrics || []).length && ["Value", "Momentum"].forEach(grp => {
  const rows = D.other_metrics.filter(m => m.cat === grp);
  if (!rows.length) return;
  app.push(H2(`${grp} metrics (add-on candidates)`));
  app.push(table(["Metric", "Definition and formula", "Group", "Better", "Filled", "Full IC", "t"], rows.map(orow),
    [1900, 7210, 1250, 800, 760, 880, 880], { align: [L, L, L, L, R, R, R], size: 16 }));
});
app.push(P("Value and momentum IC columns are single-metric results over the full sample (reports/factor_ic.py), in the configured direction."));
app.push(H2("Reversed Growth metrics"));
app.push(table(["Metric", "Definition", "Train IC", "Train t", "Test IC", "Test t"],
  D.reversed.map(r => [r.name, `${r.name.slice(1)} with higher values ranked better (actual ranks flipped; filled ranks kept)`, f3(r.train_ic), f2(r.train_t), f3(r.test_ic), f2(r.test_t)]),
  [1900, 7050, 1000, 900, 1000, 1830], { align: [L, L, R, R, R, R], size: 16 }));
if ((NOTES.dev_log || []).length) {
  app.push(H1("Appendix B. Development log"));
  app.push(P("Design changes, in order, with the reason for each:"));
  app.push(table(["Change", "Reason"], NOTES.dev_log, [6900, 6780], { size: 16 }));
}
// ---------- document ----------
const footer = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [`${SEC} Quality + Growth models · methodology · page `, PageNumber.CURRENT], size: 16, color: "6B7280" })],
  })],
});
const doc = new Document({
  creator: "Claude",
  title: `${SEC} Quality + Growth Models: Methodology`,
  styles: {
    default: { document: { run: { font: FONT, size: 21 }, paragraph: { spacing: { line: 276 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F3A5F", font: FONT }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2F5D8A", font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 320 } } } }] },
    ],
  },
  sections: [
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } }, footers: { default: footer }, children: body },
    { properties: { page: { size: { width: 12240, height: 15840, orientation: PageOrientation.LANDSCAPE }, margin: { top: 900, bottom: 900, left: 1080, right: 1080 } } }, footers: { default: footer }, children: app },
  ],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
