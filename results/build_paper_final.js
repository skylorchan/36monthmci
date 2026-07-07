// Builds ADNI_paper_FINAL.docx  — professor-revision pass on the corrected pipeline.
// Addresses Joe Xiao's 14 comments + decisions list:
//  - Clean formatting (black headings, no colored borders, no export footer/dividers)
//  - References at END; peer-reviewed journal/conference sources only
//  - Figures placed AFTER first in-text reference
//  - 3-column feature-definition table; analyses-overview table
//  - Limitations moved to end of Discussion; explicit n=308 limitation
//  - First-mention terms spelled out in full; scikit-learn plumbing trimmed
//  - Corrected honest numbers (0.862 / 0.881), NOT the buggy 0.932
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, Footer, PageNumber,
} = require("docx");
const fs = require("fs");
const path = require("path");

const DIR = "C:\\Users\\skylo\\Documents\\adni-mci-conversion\\results";
const CONTENT_W = 9360;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 140 },
    children: [new TextRun({ text, bold: true, size: 30, font: "Times New Roman", color: "000000" })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, bold: true, size: 26, font: "Times New Roman", color: "000000" })] });
}
function body(runs, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160, line: 276 }, ...opts,
    children: Array.isArray(runs) ? runs : [new TextRun({ text: runs, size: 24, font: "Times New Roman" })] });
}
function run(text, opts = {}) { return new TextRun({ text, size: 24, font: "Times New Roman", ...opts }); }
function bold(text) { return run(text, { bold: true }); }
function italic(text) { return run(text, { italics: true }); }

const bdr = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: bdr, bottom: bdr, left: bdr, right: bdr };
const hdrShade  = { fill: "ECECEC", type: ShadingType.CLEAR };   // neutral gray, not blue
const altShade  = { fill: "F6F6F6", type: ShadingType.CLEAR };
const warnShade = { fill: "F2F2F2", type: ShadingType.CLEAR };
const white = { fill: "FFFFFF", type: ShadingType.CLEAR };

function cell(text, w, { shade, isBold, center, color, italics } = {}) {
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA }, shading: shade || white,
    margins: { top: 70, bottom: 70, left: 120, right: 120 },
    children: [new Paragraph({ alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), size: 21, font: "Times New Roman", bold: !!isBold, italics: !!italics, color: color || "000000" })] })] });
}
function caption(text) {
  return new Paragraph({ spacing: { before: 80, after: 240 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, italics: true, size: 20, font: "Times New Roman", color: "333333" })] });
}
function figure(file, widthPx, heightPx) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(DIR, file)),
      transformation: { width: widthPx, height: heightPx } })] });
}
function figCaption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 240 },
    children: [new TextRun({ text, italics: true, size: 20, font: "Times New Roman", color: "333333" })] });
}

// ---------------------------------------------------------------------------
// Title
// ---------------------------------------------------------------------------
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 160 },
    children: [new TextRun({ text: "Leakage-Audited Multimodal Prediction of Mild Cognitive Impairment to Alzheimer’s Disease Conversion: An Honest Benchmark on ADNI", bold: true, size: 34, font: "Times New Roman" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 },
    children: [new TextRun({ text: "Skylor Chan", size: 26, font: "Times New Roman" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 },
    children: [new TextRun({ text: "Target venue: Journal of Engineering Innovation (JEI) / medRxiv preprint", size: 20, font: "Times New Roman", italics: true })] }),
];

// ---------------------------------------------------------------------------
// Abstract
// ---------------------------------------------------------------------------
const abstractBlock = [
  h1("Abstract"),
  body([ bold("Background. "),
    run("Predicting conversion from mild cognitive impairment (MCI) to Alzheimer’s disease (AD) is clinically valuable and methodologically contested. Published machine-learning models report values of the area under the receiver-operating-characteristic curve (AUC) ranging from 0.70 to above 0.99 on the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset, yet most results are inflated by data leakage, particularly failure to enforce subject-level train-test separation and application of preprocessing steps across the split boundary.") ]),
  body([ bold("Methods. "),
    run("We built a reproducible, leakage-free pipeline predicting 36-month MCI-to-AD conversion from baseline multimodal ADNI features (cognitive assessments, magnetic resonance imaging (MRI) volumetrics, amyloid positron emission tomography (PET), cerebrospinal fluid (CSF) biomarkers, apolipoprotein E (APOE) genotype, and demographics) and 12-month longitudinal slopes in a cohort of 308 subjects (149 converters / 159 non-converters). Subject-level train-test separation was enforced throughout, with preprocessing (median imputation, tabular augmentation) applied strictly within cross-validation folds using pipeline objects that structurally prevent fitting on held-out data. We conducted (1) a leakage-tax ablation quantifying AUC inflation from two common leakage patterns, (2) a split-stability analysis across 200 random subject-level splits, (3) calibration analysis, (4) decision curve analysis, and (5) a missing-modality robustness benchmark.") ]),
  body([ bold("Results. "),
    run("Across 200 random subject-level splits the clean pipeline achieved a mean test AUC of 0.862 (95% range 0.785–0.942), a mean cross-validation AUC of 0.861, and a mean Brier score of 0.154; a single pre-specified held-out split gave a test AUC of 0.881 (95% confidence interval 0.788–0.956) with a Brier score of 0.145. Augmenting training data before the split inflated cross-validation AUC from 0.821 to 0.980 (+0.159), a result a leaky study would publish as its headline number. Imputing on combined train and test data had a negligible effect. Dropping any single expensive modality (amyloid PET, CSF, MRI) reduced AUC by at most 0.012 (within noise), suggesting that cognitive scores and APOE status capture most prognostic signal. The model demonstrated net clinical benefit over treat-all and treat-none strategies across decision thresholds from 5% to 85%.") ]),
  body([ bold("Conclusions. "),
    run("Design choices which lower the headline AUC, namely subject-level splitting, preprocessing inside cross-validation folds, and honest augmentation, are precisely those that make the number trustworthy, and reporting a distribution over many splits rather than a single value is part of the same discipline. We release the full pipeline as a reproducible open-source artifact. The finding that cognition-only models closely match full-modality performance has practical implications for clinical settings where PET and CSF are unavailable.") ]),
  body([ bold("Keywords: "),
    italic("mild cognitive impairment, Alzheimer’s disease, machine learning, data leakage, gradient boosting, ADNI, calibration, decision curve analysis") ]),
];

// ---------------------------------------------------------------------------
// 1. Introduction
// ---------------------------------------------------------------------------
const introBlock = [
  h1("1. Introduction"),
  body("Alzheimer’s disease (AD) is the leading cause of dementia worldwide, affecting an estimated 55 million people and placing enormous burden on patients, caregivers, and health systems [1]. Mild cognitive impairment (MCI) is a transitional state between normal aging and dementia in which patients exhibit objectively measured cognitive deficits that do not yet impair daily function [2, 3]. Approximately 10–15% of MCI patients progress to AD annually, but individual trajectories are highly variable: some patients convert within months while others remain stable for decades [4]. Accurate prediction of who will convert within a clinically actionable window, typically 24–36 months [4], would enable targeted enrolment in prevention trials and earlier initiation of emerging disease-modifying therapies."),
  body("The Alzheimer’s Disease Neuroimaging Initiative (ADNI) has assembled one of the largest longitudinal multimodal datasets for this purpose [5], and dozens of machine-learning studies have applied it to the MCI conversion task [6]. Reported AUC values in the literature span a remarkable range, from honest estimates near 0.72–0.82 [7] to values above 0.95, a spread documented in a recent systematic review [6]. This heterogeneity is not explained by genuine differences in predictive signal; it is explained, in large part, by data leakage, meaning systematic violations of the data-generating process during model development that make performance estimates overly optimistic [8]."),
  body("Kapoor and Narayanan (2023) [8] audited the clinical-prediction-model literature and identified a taxonomy of common leakage patterns: (1) failure to enforce subject-level train-test splitting when multiple records exist per subject; (2) applying preprocessing (imputation, scaling, oversampling) to the full dataset before splitting, allowing test-set statistics to contaminate training; (3) performing hyperparameter search on the test set; and (4) including post-baseline outcome information in the feature vector. Each pattern alone can add 0.05–0.20 AUC to reported results on small cohorts."),
  body("This paper makes three contributions:"),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "contributions", level: 0 },
    children: [bold("Leakage-audited benchmark."), run(" We quantify the AUC inflation attributable to two specific leakage patterns on the same ADNI cohort and seed, providing a concrete, reproducible demonstration of the Kapoor-Narayanan taxonomy applied to MCI conversion prediction.")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "contributions", level: 0 },
    children: [bold("Reproducible, rigorously validated pipeline."), run(" We refactor the entire analysis into a modular open-source repository with structural leakage guards (pipeline objects enforcing preprocessing inside folds), automated leakage-check tests, and a single command to reproduce all results from raw ADNI data.")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160, line: 276 }, numbering: { reference: "contributions", level: 0 },
    children: [bold("Clinically interpretable finding."), run(" Feature-attribution analysis demonstrates that early longitudinal cognitive change (the 12-month slope of the Alzheimer’s Disease Assessment Scale-Cognitive subscale, ADAS-Cog) combined with baseline severity (the Clinical Dementia Rating Sum of Boxes, CDR-SB, and the Mini-Mental State Examination, MMSE) and apolipoprotein E ε4 (APOEε4) status accounts for most predictive signal, with an important corollary: dropping expensive imaging modalities (MRI, amyloid PET, CSF) costs at most 0.012 AUC, a difference indistinguishable from noise at this cohort size. This directly addresses a practical clinical question: do patients need an amyloid PET scan to be accurately risk-stratified?")] }),
  body("Critically, the honest validated AUC of this pipeline is not the highest in the literature; it is mid-pack. The value of this work is not the number; it is the demonstration that the number can be trusted. We argue that a well-calibrated, leakage-free model with AUC 0.86 is more useful to a clinician than an inflated model claiming 0.97, and we provide decision curve analysis to support that argument in terms of net clinical benefit."),
];

// ---------------------------------------------------------------------------
// 2. Methods
// ---------------------------------------------------------------------------
const methodsBlock = [
  h1("2. Methods"),
  h2("2.1 Data Source"),
  body("Data were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI; adni.loni.usc.edu). ADNI was launched in 2003 as a public-private partnership to test whether serial MRI, PET, biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of MCI and early AD [5]. Access was obtained through the standard ADNI application process. All data used here were downloaded as of February 2026 and are governed by the ADNI Data Use Agreement. The tables used and the features derived from each are listed in Table 1."),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [2400, 2800, 4160],
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("ADNI table", 2400, { shade: hdrShade, isBold: true }),
        cell("Contents", 2800, { shade: hdrShade, isBold: true }),
        cell("Features derived", 4160, { shade: hdrShade, isBold: true }) ]}),
      ...[
        ["DXSUM_PDXCONV", "Diagnosis per visit (cognitively normal, MCI, or AD)", "Baseline MCI identification; 36-month conversion label"],
        ["PTDEMOG", "Demographics (static)", "Age, sex, education, site"],
        ["APOERES", "APOE genotype", "APOEε4 allele count (0/1/2) derived from the genotype field"],
        ["MMSE", "Mini-Mental State Examination", "Baseline score; 12-month change"],
        ["CDR", "Clinical Dementia Rating", "CDR Sum of Boxes (CDR-SB); baseline and 12-month change"],
        ["ADAS_ADNI1", "Alzheimer’s Disease Assessment Scale", "ADAS-Cog total score; baseline and 12-month change"],
        ["NEUROBAT", "Neuropsychological battery", "Supplementary cognitive measures"],
        ["UCSFFSX7", "FreeSurfer 7 MRI volumetrics", "Hippocampal, entorhinal, and ventricular volumes (normalised by intracranial volume); 12-month changes"],
        ["UCBERKELEY_AMY_6MM", "Amyloid PET (florbetapir)", "Summary standardized uptake value ratio (35% missing); amyloid positivity status"],
        ["UPENNBIOMK_ROCHE_ELECSYS", "CSF biomarkers", "Amyloid-beta 42, total tau, phosphorylated tau-181 (45% missing)"],
      ].map(([tbl, contents, feats], i) =>
        new TableRow({ children: [
          cell(tbl, 2400, { shade: i % 2 === 0 ? white : altShade }),
          cell(contents, 2800, { shade: i % 2 === 0 ? white : altShade }),
          cell(feats, 4160, { shade: i % 2 === 0 ? white : altShade }) ]}) ),
    ] }),
  caption("Table 1. ADNI tables used and the features derived from each."),

  h2("2.2 Cohort Construction"),
  body("Starting from all subjects with at least one MCI diagnosis in DXSUM_PDXCONV (n = 417), we identified each subject’s index visit as the first recorded MCI diagnosis. Subjects were retained if they met all of the following criteria:"),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "criteria", level: 0 }, children: [run("At least 2 recorded visits after the index visit.")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "criteria", level: 0 }, children: [run("At least 36 months of follow-up from the index visit, OR a confirmed AD diagnosis within 36 months (ensuring all converters are captured regardless of total follow-up duration).")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160, line: 276 }, numbering: { reference: "criteria", level: 0 }, children: [run("At least 12 months of follow-up, OR a confirmed AD diagnosis within 36 months (required for 12-month slope feature derivation).")] }),
  body("These criteria yielded a final cohort of 308 subjects (149 converters, 159 non-converters). The exclusion flow is shown in Figure 1."),
  figure("censoring_flow.png", 430, 482),
  figCaption("Figure 1. Consolidated Standards of Reporting Trials (CONSORT)-style subject inclusion flow. Of 417 subjects with at least one MCI visit, 308 were retained after applying the visit-count and follow-up criteria; 109 (26%) were excluded as right-censored."),

  h2("2.3 Outcome Definition"),
  body([ run("The primary outcome was binary 36-month conversion: label = 1 if a subject received an AD diagnosis at any visit within 36 months of the index MCI visit, and label = 0 if the subject had at least 36 months of follow-up without receiving an AD diagnosis. Subjects with less than 36 months of follow-up who did not convert were treated as censored and excluded from the primary analysis (n = 109; 26% of the pre-filtered cohort). The implications of this censoring strategy are discussed in the Limitations "), italic("(Section 4.1)"), run(".") ]),

  h2("2.4 Feature Engineering"),
  body("Feature extraction followed a two-stage process, using only data available at or before 15 months from the index visit to prevent any information from the outcome window entering the feature vector. Baseline features took the record closest to the index visit for each subject; MRI volumes were normalised by intracranial volume (ICV) to control for head size. To capture early disease trajectory, 12-month change scores were computed for CDR-SB, MMSE, ADAS-Cog, and hippocampal volume, using the visit falling within a 9-to-15-month window from the index visit. For every feature with any missing values, a binary indicator variable was appended so the model could learn whether the absence of a measurement is itself informative. The final feature matrix comprised 25 base features and 9 missingness indicators (34 features total). All features are documented in Table 2."),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [2700, 1860, 4800],
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("Feature", 2700, { shade: hdrShade, isBold: true }),
        cell("Variable type", 1860, { shade: hdrShade, isBold: true }),
        cell("Definition", 4800, { shade: hdrShade, isBold: true }) ]}),
      ...[
        ["CDR-SB", "Continuous", "Clinical Dementia Rating Sum of Boxes at baseline (0–18; higher is more impaired)"],
        ["MMSE", "Continuous", "Mini-Mental State Examination total at baseline (0–30; lower is worse)"],
        ["ADAS-Cog (13)", "Continuous", "Alzheimer’s Disease Assessment Scale-Cognitive subscale, 13-item total at baseline"],
        ["Hippocampus L / R", "Continuous", "Left and right hippocampal volume (mm³)"],
        ["Entorhinal L / R", "Continuous", "Left and right entorhinal cortex volume (mm³)"],
        ["Ventricles", "Continuous", "Total ventricular volume (mm³)"],
        ["Hippocampus L / R (norm.)", "Continuous", "Left and right hippocampal volume normalised by intracranial volume"],
        ["Ventricles (norm.)", "Continuous", "Ventricular volume normalised by intracranial volume"],
        ["Amyloid SUVR", "Continuous", "Amyloid PET summary standardized uptake value ratio (florbetapir)"],
        ["Aβ42", "Continuous", "CSF amyloid-beta 42 concentration (pg/mL)"],
        ["Total tau", "Continuous", "CSF total tau concentration (pg/mL)"],
        ["Phospho-tau-181", "Continuous", "CSF phosphorylated tau-181 concentration (pg/mL)"],
        ["APOEε4 count", "Categorical (0–2)", "Number of APOEε4 alleles"],
        ["Sex", "Categorical (binary)", "Biological sex"],
        ["Education", "Continuous", "Years of formal education"],
        ["Ethnicity", "Categorical", "Ethnicity category"],
        ["Race", "Categorical", "Race category"],
        ["Δ CDR-SB, Δ MMSE, Δ ADAS-Cog", "Continuous", "12-month change in each cognitive score"],
        ["Δ Hippocampus L / R (norm.)", "Continuous", "12-month change in normalised hippocampal volume"],
        ["Missingness indicators (×9)", "Categorical (binary)", "One flag per feature with missing values, set to 1 when that measurement is absent"],
      ].map(([f, t, d], i) =>
        new TableRow({ children: [
          cell(f, 2700, { shade: i % 2 === 0 ? white : altShade }),
          cell(t, 1860, { shade: i % 2 === 0 ? white : altShade }),
          cell(d, 4800, { shade: i % 2 === 0 ? white : altShade }) ]}) ),
    ] }),
  caption("Table 2. Feature definitions. Continuous features are real-valued measurements; categorical features are discrete codes. The nine missingness indicators are collapsed into a single row for brevity."),

  h2("2.5 Subject-Level Train-Test Split"),
  body([ run("A "), bold("subject-level"), run(" split was enforced throughout: all records belonging to a given subject (identified by their unique ADNI subject identifier) were assigned exclusively to either the training set or the test set. The pre-specified held-out test set comprised 20% of subjects, stratified by conversion status (n = 61; 30 converters / 31 non-converters), fixed at a single random seed. Because a split of this size yields a fragile estimate, we additionally repeated the split over 200 random seeds (Section 2.8). Within the training set, a subject-grouped 5-fold cross-validation was used, again ensuring no subject appeared in both the training and validation partitions of any fold.") ]),

  h2("2.6 Preprocessing and Model"),
  body("All preprocessing was applied within each cross-validation fold, using pipeline objects that structurally prevent preprocessing steps from fitting on validation or test data. Each fold applied median imputation fit on the training fold only, followed by tabular augmentation of the training data (two noisy copies per record: Gaussian noise scaled to 1.5% of each feature’s standard deviation, plus 5% of non-missing values randomly hidden to simulate realistic missingness). Validation and test partitions were always clean, unaugmented data. The classifier was a gradient-boosted tree ensemble [10], which handles missing inputs natively by learning the optimal branch direction for absent values."),

  h2("2.7 Hyperparameter Optimisation"),
  body("Hyperparameters were tuned over 80 trials of Bayesian optimisation (a tree-structured Parzen estimator), maximising mean AUC across the 5-fold grouped cross-validation on the training set. The search covered the number of trees, tree depth, learning rate, subsampling rates, minimum child weight, and regularisation strengths. All hyperparameter search used only training-set data; the test set was never accessed during optimisation, and the tuned configuration was then held fixed for the split-stability analysis."),

  h2("2.8 Evaluation and Analyses"),
  body("The final model was trained on the full augmented training set with the tuned hyperparameters, then evaluated on the held-out test set. Discrimination was summarised by AUC and the area under the precision-recall curve (AUPRC), each with 2,000-replicate bootstrap 95% confidence intervals; calibration by the Brier score and a reliability diagram; clinical utility by decision curve analysis [12]; and feature importance by SHapley Additive exPlanations (SHAP) values [11]. Table 3 summarises the analyses conducted."),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [2760, 3900, 2700],
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("Analysis", 2760, { shade: hdrShade, isBold: true }),
        cell("Purpose", 3900, { shade: hdrShade, isBold: true }),
        cell("Primary output", 2700, { shade: hdrShade, isBold: true }) ]}),
      ...[
        ["Main model performance", "Discrimination and calibration on held-out data", "Table 4"],
        ["Split-stability analysis", "Sensitivity of the estimate to the choice of split (200 splits)", "Figure 2"],
        ["Leakage-tax ablation", "AUC inflation from two common leakage patterns", "Table 5, Figure 5"],
        ["Calibration analysis", "Whether predicted probabilities can be trusted", "Figure 3"],
        ["Decision curve analysis", "Net clinical benefit vs treat-all / treat-none", "Figure 4"],
        ["Missing-modality benchmark", "Marginal value of each imaging / biomarker modality", "Table 6, Figure 6"],
        ["Feature importance", "Which features drive the predictions", "Figure 7"],
      ].map(([a, p, o], i) =>
        new TableRow({ children: [
          cell(a, 2760, { shade: i % 2 === 0 ? white : altShade }),
          cell(p, 3900, { shade: i % 2 === 0 ? white : altShade }),
          cell(o, 2700, { shade: i % 2 === 0 ? white : altShade, center: true }) ]}) ),
    ] }),
  caption("Table 3. Overview of the analyses conducted and where each is reported."),
  body([ bold("Split-stability analysis. "), run("Because a single held-out test set of n = 61 yields a fragile point estimate, we repeated the entire split-train-evaluate procedure across 200 random stratified subject-level splits with the tuned hyperparameters held fixed, and report the distribution of cross-validation AUC, test AUC, test AUPRC, and Brier score. This isolates test-draw variance for a fixed model configuration and is the primary summary of performance.") ]),
  body([ bold("Leakage-tax ablation. "), run("To quantify the AUC inflation attributable to common leakage patterns, we ran three additional pipeline variants on the same cohort and seed: (A) augmentation applied before the train-test split combined with a record-level (non-subject-grouped) random split; (B) median imputation fit on the combined train and test dataset before cross-validation; and (C) both patterns simultaneously. All variants used identical hyperparameters.") ]),
  body([ bold("Missing-modality benchmark. "), run("To assess the marginal value of each imaging and biomarker modality, we performed a systematic ablation across six modality configurations: full model, drop amyloid PET, drop CSF, drop MRI, drop all three (cognition and demographics only), and cognition only. For each configuration, 5-fold grouped cross-validation AUC was reported for both native missing-value handling and median imputation.") ]),

  h2("2.9 Software and Reproducibility"),
  body("All analyses were implemented in Python. The full pipeline is available at https://github.com/skylorchan/36monthmci, and a single command reproduces all figures and tables from the raw ADNI data. Leakage guard tests assert that no subject appears in both training and test partitions of any split, that the split is stratified and deterministic, and that imputers are fit only on training data. During development these checks caught an index-alignment error in an earlier version of the split-reconstruction code that had produced a positional, non-stratified hold-out set and an over-optimistic test AUC of 0.93; all results reported here use the corrected, stratified subject-level split, and the discrepancy is discussed in Section 4."),
];

// ---------------------------------------------------------------------------
// 3. Results
// ---------------------------------------------------------------------------
const resultsBlock = [
  h1("3. Results"),
  h2("3.1 Cohort Characteristics"),
  body("Starting from 417 subjects with at least one MCI visit in ADNI, 308 met all inclusion criteria (149 converters, 159 stable non-converters; conversion rate 48.4%). The primary reason for exclusion was insufficient follow-up: 92 subjects had less than 36 months of recorded follow-up without a confirmed AD diagnosis (right-censored). An additional 17 subjects had fewer than 2 visits and were excluded because slope features could not be derived. Mean follow-up in the retained cohort was 64.5 months (standard deviation 42.6); censored subjects had a mean follow-up of 17.1 months. The implications of this asymmetry are addressed in the Limitations. Each pre-specified held-out test set comprised 61 subjects, stratified to match the cohort conversion prevalence of roughly 49%."),

  h2("3.2 Main Model Performance and Split Stability"),
  body("The tuned gradient-boosted model (80 tuning trials, 5-fold subject-grouped cross-validation) achieved a cross-validation AUC of 0.837 on the primary training split. Because a single held-out test set of n = 61 yields a fragile point estimate, we summarise performance both as a distribution over 200 random subject-level splits and as a single pre-specified hold-out split (Table 4, Figure 2)."),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [2760, 3300, 3300],
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("Metric", 2760, { shade: hdrShade, isBold: true }),
        cell("200 random splits: mean [95% range]", 3300, { shade: hdrShade, isBold: true, center: true }),
        cell("Primary single split [95% CI]", 3300, { shade: hdrShade, isBold: true, center: true }) ]}),
      new TableRow({ children: [ cell("5-fold cross-validation AUC", 2760), cell("0.861  [0.830 – 0.887]", 3300, { center: true }), cell("0.837", 3300, { center: true }) ]}),
      new TableRow({ children: [ cell("Test AUC", 2760, { shade: altShade }), cell("0.862  [0.785 – 0.942]", 3300, { shade: altShade, center: true, isBold: true }), cell("0.881  [0.788 – 0.956]", 3300, { shade: altShade, center: true, isBold: true }) ]}),
      new TableRow({ children: [ cell("Test AUPRC", 2760), cell("0.861  [0.784 – 0.938]", 3300, { center: true }), cell("0.896  [0.799 – 0.963]", 3300, { center: true }) ]}),
      new TableRow({ children: [ cell("Test Brier score", 2760, { shade: altShade }), cell("0.154  [0.110 – 0.191]", 3300, { shade: altShade, center: true }), cell("0.145", 3300, { shade: altShade, center: true }) ]}),
    ] }),
  caption("Table 4. Split-robust performance. Left: distribution across 200 random stratified subject-level splits with the tuned model held fixed (test n = 61 each). Right: the pre-specified single hold-out split (30 converters), with 2,000-replicate bootstrap confidence intervals. The single-split test AUC (0.881) is the 68th percentile of the distribution."),
  figure("repeated_splits.png", 520, 316),
  figCaption("Figure 2. Split-stability analysis. Distribution of held-out test AUC across 200 random subject-level splits with the tuned model held fixed (test n = 61 each). Mean 0.862, 95% range [0.785, 0.942]. The dashed line marks the pre-specified single split (0.881, 68th percentile)."),
  body("Across the 200 splits the test AUC was stable (mean 0.862, standard deviation 0.039, 95% range 0.785–0.942), and the cross-validation AUC was tighter still (0.861, 95% range 0.830–0.887). The pre-specified single split gave a test AUC of 0.881, at the 68th percentile of the distribution: a mildly favourable but representative draw. We treat the 200-split range, not any single number, as the honest summary of model performance. Even across many splits the 95% range spans roughly 0.16 AUC, reflecting the small test set (n = 61); a larger study with several hundred test subjects would narrow this substantially."),
  body([ bold("Calibration. "), run("The reliability diagram (Figure 3) shows that predicted probabilities are reasonably well-calibrated, with observed conversion fractions tracking the mean predicted probability across quantile-binned groups. The single-split Brier score of 0.145 compares favourably to a naive prevalence-based predictor (Brier = prevalence × (1 − prevalence) = 0.250), a 42% reduction in mean squared error.") ]),
  figure("calibration.png", 400, 457),
  figCaption("Figure 3. Calibration (reliability diagram). Observed conversion fraction versus mean predicted probability across quantile-binned groups, with a histogram of predicted probabilities below. Brier score = 0.145."),
  body([ bold("Clinical utility. "), run("Decision curve analysis (Figure 4) demonstrated that the model provides net clinical benefit over both the treat-all and treat-none strategies across the full range of decision thresholds from 5% to 85%. At any clinically plausible threshold for initiating preventive intervention or trial enrolment, acting on the model’s predictions leads to better expected outcomes, in terms of true positives gained minus false positives weighted by their relative harm, than either treating every MCI patient or treating none.") ]),
  figure("dca.png", 520, 345),
  figCaption("Figure 4. Decision curve analysis. Net benefit of the model versus treat-all and treat-none strategies across decision thresholds from 5% to 85%. The model provides positive net benefit across the full clinically plausible range."),

  h2("3.3 Leakage-Tax Ablation"),
  body("To quantify the AUC inflation attributable to common leakage patterns, we ran three additional pipeline variants on the same cohort and seed using fixed (non-tuned) hyperparameters, allowing differences to be attributed to the leakage design alone. Results are shown in Table 5 and visualised in Figure 5."),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [3600, 1440, 1440, 1440, 1440],
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("Pipeline", 3600, { shade: hdrShade, isBold: true }),
        cell("CV AUC", 1440, { shade: hdrShade, isBold: true, center: true }),
        cell("Test AUC", 1440, { shade: hdrShade, isBold: true, center: true }),
        cell("AUPRC", 1440, { shade: hdrShade, isBold: true, center: true }),
        cell("Brier", 1440, { shade: hdrShade, isBold: true, center: true }) ]}),
      new TableRow({ children: [
        cell("Clean (subject split, preprocessing inside folds)", 3600),
        cell("0.821", 1440, { center: true }),
        cell("0.853", 1440, { center: true, isBold: true }),
        cell("0.857", 1440, { center: true }),
        cell("0.178", 1440, { center: true }) ]}),
      new TableRow({ children: [
        cell("Leak A: augment before split + record-level split", 3600, { shade: warnShade }),
        cell("0.980", 1440, { shade: warnShade, center: true, isBold: true }),
        cell("0.976", 1440, { shade: warnShade, center: true }),
        cell("0.974", 1440, { shade: warnShade, center: true }),
        cell("0.064", 1440, { shade: warnShade, center: true }) ]}),
      new TableRow({ children: [
        cell("Leak B: impute on combined train + test before CV", 3600),
        cell("0.819", 1440, { center: true }),
        cell("0.856", 1440, { center: true }),
        cell("0.856", 1440, { center: true }),
        cell("0.176", 1440, { center: true }) ]}),
      new TableRow({ children: [
        cell("Leak A + B: both patterns combined", 3600, { shade: warnShade }),
        cell("0.978", 1440, { shade: warnShade, center: true, isBold: true }),
        cell("0.964", 1440, { shade: warnShade, center: true }),
        cell("0.963", 1440, { shade: warnShade, center: true }),
        cell("0.075", 1440, { shade: warnShade, center: true }) ]}),
    ] }),
  caption("Table 5. Leakage-tax ablation. All variants use identical hyperparameters and the same cohort and seed. Shaded rows are leaky pipelines. The cross-validation (CV) AUC is what a leaky paper would report; the test AUC is the honest held-out result."),
  figure("leakage_tax.png", 580, 245),
  figCaption("Figure 5. Leakage-tax ablation. Cross-validation AUC for the clean pipeline versus three leaky variants on the same cohort and seed. Augmenting before the split (Leak A) inflates cross-validation AUC from 0.821 to 0.980, an increase of 0.159."),
  body([ bold("Leak A (augmentation before split with a record-level split)"), run(" was the most damaging: cross-validation AUC inflated from 0.821 to 0.980, a +0.159 increase that would be published as the headline result in a leaky study. The mechanism is direct contamination: augmented (noisy) copies of test subjects appear in the training set, and because the split is record-level rather than subject-level, the model effectively memorises distorted versions of its own test cases during cross-validation. This maps directly to the Kapoor-Narayanan (2023) first leakage type.") ]),
  body([ bold("Leak B (imputation on full data)"), run(" had a negligible effect in this cohort (−0.002 cross-validation AUC), because the model is relatively robust to small shifts in median imputation statistics when features are continuous and cohort-wide distributions are smooth. This leak would be more damaging in smaller cohorts or when imputing high-missingness features (amyloid PET, 35% missing; CSF, 45% missing).") ]),
  body("Together, these results demonstrate that a researcher following common but incorrect practices would report a cross-validation AUC of 0.978–0.980 on this dataset, numbers that would rank at the upper end of the ADNI literature, while the honest held-out test AUC of the clean model averages 0.862."),

  h2("3.4 Missing-Modality Robustness"),
  body("Table 6 reports 5-fold grouped cross-validation AUC under systematic modality ablation, illustrated in Figure 6. Each row represents a model trained and evaluated using only the listed feature subsets, comparing native missing-value handling to median imputation."),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [3360, 960, 1440, 1440, 1440, 660],
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("Modality configuration", 3360, { shade: hdrShade, isBold: true }),
        cell("Feat.", 960, { shade: hdrShade, isBold: true, center: true }),
        cell("AUC (native)", 1440, { shade: hdrShade, isBold: true, center: true }),
        cell("AUC (imputed)", 1440, { shade: hdrShade, isBold: true, center: true }),
        cell("AUPRC", 1440, { shade: hdrShade, isBold: true, center: true }),
        cell("Δ vs full", 660, { shade: hdrShade, isBold: true, center: true }) ]}),
      ...[
        ["Full model (all modalities)",               "25", "0.824 ± 0.056", "0.821 ± 0.066", "0.818", "ref"],
        ["Drop amyloid PET",                          "24", "0.819 ± 0.073", "0.816 ± 0.074", "0.805", "-0.005"],
        ["Drop CSF",                                  "22", "0.833 ± 0.051", "0.826 ± 0.055", "0.825", "+0.009"],
        ["Drop MRI",                                  "15", "0.824 ± 0.055", "0.813 ± 0.073", "0.812", "0.000"],
        ["Drop MRI + PET + CSF (cognition + demo)",   "11", "0.821 ± 0.063", "0.823 ± 0.059", "0.812", "-0.003"],
        ["Cognition only",                             "6", "0.812 ± 0.076", "0.811 ± 0.077", "0.808", "-0.012"],
      ].map(([name, n, nat, imp, auprc, delta], i) =>
        new TableRow({ children: [
          cell(name, 3360, { shade: i % 2 === 0 ? white : altShade }),
          cell(n, 960, { shade: i % 2 === 0 ? white : altShade, center: true }),
          cell(nat, 1440, { shade: i % 2 === 0 ? white : altShade, center: true }),
          cell(imp, 1440, { shade: i % 2 === 0 ? white : altShade, center: true }),
          cell(auprc, 1440, { shade: i % 2 === 0 ? white : altShade, center: true }),
          cell(delta, 660, { shade: i % 2 === 0 ? white : altShade, center: true }) ]}) ),
    ] }),
  caption("Table 6. Missing-modality robustness benchmark (5-fold grouped cross-validation on the training set). Δ = AUC (native handling) minus full-model AUC. All differences are within one standard deviation and should be interpreted cautiously given overlapping confidence intervals."),
  figure("missing_modality.png", 580, 247),
  figCaption("Figure 6. Missing-modality robustness. Cross-validation AUC under systematic modality ablation. Dropping any single expensive modality (amyloid PET, CSF, or MRI) changes AUC by at most 0.012, smaller than the standard deviation of any single fold."),
  body("The central finding is that dropping any single expensive modality, whether amyloid PET, CSF biomarkers, or MRI, reduces AUC by at most 0.012, a difference smaller than the standard deviation of any individual fold. Even the cognition-only model (CDR-SB, MMSE, ADAS-Cog and their 12-month slopes; 6 features) achieves AUC 0.812 ± 0.076 versus 0.824 ± 0.056 for the full 25-feature model. These differences are statistically indistinguishable at n = 308. Native missing-value handling matched or slightly outperformed median imputation in all configurations, so explicit imputation provides no benefit here."),

  h2("3.5 Feature Importance"),
  body("Global SHAP values computed on the imputed test set identified the following top predictors (Figure 7), consistent with clinical expectations:"),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "shap", level: 0 },
    children: [bold("CDR-SB at baseline"), run(": the single largest contributor. Higher baseline severity strongly predicts conversion.")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "shap", level: 0 },
    children: [bold("MMSE at baseline"), run(": lower baseline MMSE increases conversion probability, capturing global cognitive status.")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "shap", level: 0 },
    children: [bold("12-month ADAS-Cog slope"), run(": the rate of early cognitive decline, not just baseline severity. Subjects who worsen over the first year are substantially more likely to convert, reinforcing the value of the 12-month follow-up window as a prognostic signal.")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 }, numbering: { reference: "shap", level: 0 },
    children: [bold("APOEε4 allele count"), run(": consistent with its established role as the strongest genetic risk factor for late-onset AD [9].")] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 200, line: 276 }, numbering: { reference: "shap", level: 0 },
    children: [bold("Left hippocampal volume"), run(": a smaller hippocampus at baseline is associated with higher conversion risk, consistent with early entorhinal-hippocampal neurodegeneration in AD.")] }),
  figure("shap_summary.png", 400, 481),
  figCaption("Figure 7. SHAP feature importance (beeswarm). Global mean absolute SHAP values on the test set. Cognitive severity (CDR-SB, MMSE), early cognitive decline (the 12-month ADAS-Cog slope), and APOEε4 status dominate the ranking."),
  body("Notably, all top five features are either cognitive assessments or a structural MRI measure, with amyloid PET and CSF biomarkers ranking lower. This is consistent with the missing-modality benchmark finding that imaging biomarkers add little incremental discriminative value in this cohort, likely because amyloid PET is missing for 65% of subjects and CSF for 45%."),
];

// ---------------------------------------------------------------------------
// 4. Discussion
// ---------------------------------------------------------------------------
const discussionBlock = [
  h1("4. Discussion"),
  body("This study presents a leakage-audited, reproducible pipeline for 36-month MCI-to-AD conversion prediction on ADNI. Across 200 random subject-level splits the model achieved a mean test AUC of 0.862 (95% range 0.785–0.942), with a pre-specified single split at 0.881 (95% CI 0.788–0.956) and a Brier score of 0.145. Performance is reported as a distribution rather than a single number because the small test set (n = 61) makes any individual point estimate fragile: values from roughly 0.79 to 0.94 are consistent with the data. This transparency is itself a contribution."),
  body([ bold("The leakage tax is large and reproducible."), run(" The most striking result of this study is not the AUC; it is the demonstration that augmenting training data before the train-test split inflated cross-validation AUC by 0.159 (from 0.821 to 0.980) on the same cohort and seed. A researcher who made this single design error would publish a headline cross-validation AUC of 0.980 while the clean model’s honest held-out AUC averages 0.862. The Kapoor-Narayanan (2023) [8] taxonomy predicts this class of error, and this work provides a concrete, fully reproducible instance of it in the MCI conversion literature.") ]),
  body([ bold("Evaluation bugs are subtle, and redundancy catches them."), run(" Rigour is a process, not a checkbox. During this work we found that an earlier version of our own pipeline contained an index-alignment error that silently produced a positional, non-stratified hold-out set and reported a test AUC of 0.93. It was caught by cross-checking that single number against the 200-split distribution, into which a value that high falls only near the upper tail. We report this openly because it illustrates the paper’s thesis from the inside: even a pipeline built specifically to avoid leakage can harbour subtle evaluation bugs, and only redundant, distributional checking reliably exposes them. The corrected pipeline and its guard tests are released in full.") ]),
  body([ bold("Calibration matters as much as discrimination."), run(" The subfield has over-indexed on AUC as its primary metric. A model with AUC 0.88 but poor calibration cannot be used for individualised risk communication; a clinician cannot tell a patient that a predicted probability of 70% means a 70% chance of conversion if the model systematically over- or under-estimates in that range. The reliability diagram here shows reasonable calibration, and the decision curve demonstrates net clinical benefit across a wide threshold range. Future work in this area should routinely report the Brier score, reliability curves, and decision curve analysis alongside AUC.") ]),
  body([ bold("Cognitive assessments capture most prognostic signal."), run(" The missing-modality benchmark provides a practically important finding: dropping all three expensive modalities (amyloid PET, CSF biomarkers, and MRI) reduces AUC by only 0.003 (0.824 to 0.821), smaller than the standard deviation of any single fold. Cognition-only models achieve AUC 0.812 at a fraction of the cost and patient burden. This does not mean imaging biomarkers are clinically irrelevant, since amyloid positivity has mechanistic implications for trial eligibility and treatment selection, but it does suggest that a risk-stratification screen for clinical-trial recruitment could rely primarily on cognitive assessments without a meaningful loss of predictive accuracy.") ]),
  body([ bold("Longitudinal trajectory outperforms static baseline."), run(" The 12-month ADAS-Cog slope was the third-ranked predictor, ahead of APOEε4 and hippocampal volume. This confirms that cognitive trajectory over the first year provides signal beyond what is available at a single baseline visit. Practically, a 12-month cognitive follow-up assessment, which is standard in most ADNI-aligned clinical protocols [5], meaningfully improves risk prediction even when imaging modalities are unavailable.") ]),

  h2("4.1 Limitations"),
  body("Cohort selection introduced right-censoring bias. Of 417 subjects with at least one MCI visit in ADNI, 109 (26%) were excluded: 17 had fewer than 2 visits, and 92 lacked 36 months of follow-up without a confirmed conversion. The retained cohort (n = 308) over-represents subjects with long follow-up (mean 64.5 months versus 17.1 months for excluded subjects). Excluded subjects had significantly lower CDR-SB at baseline (1.61 vs. 2.00, Welch t-test p = 0.005), indicating the retained cohort skews toward more cognitively impaired MCI patients. The excluded subjects, those with mild MCI who dropped out before 36 months, are precisely those where discrimination is hardest, so real-world performance is likely to be lower than reported here."),
  body("The modest overall sample size is a further limitation. With 308 subjects and only 61 in each held-out test set, confidence intervals are wide (roughly 0.16 AUC), statistical power for detecting small effects is limited, and well-powered subgroup analyses are not possible; this is precisely why performance is reported as a distribution over 200 splits rather than a single value. Additional limitations are that the analysis uses a single cohort (ADNI), limiting generalisability; that no tau PET features are included; and that the APOEε4 allele count is derived from raw genotype strings rather than a curated field."),

  h2("4.2 Future Directions"),
  body("The most principled correction for censoring bias is a time-to-event framing using survival models (for example, random survival forests or gradient-boosted survival models), which recover information from censored subjects and report time-dependent AUC at 12, 24, and 36 months; this would recover all 417 subjects and eliminate the selection bias. External validation on an independent cohort such as the Open Access Series of Imaging Studies (OASIS-3) would provide the strongest evidence of generalisability and is a planned extension. Finally, the inclusion of tau PET, available in later ADNI phases but not yet incorporated here, may add discriminative signal beyond amyloid, as tau burden more directly reflects neurodegeneration."),

  h1("5. Conclusion"),
  body("We present a leakage-audited, reproducible pipeline for 36-month MCI-to-AD conversion prediction that achieves an honest test AUC of 0.862 (95% range 0.785–0.942 across 200 subject-level splits) on ADNI. The primary contribution is not the AUC; it is the demonstration that standard but incorrect practices inflate cross-validation AUC by up to 0.159 on this dataset, together with a fully reproducible artifact that others can use as a leakage-free baseline. The finding that cognitive assessments alone approach full-modality performance has practical implications for clinical risk stratification in settings where PET and CSF are unavailable. We hope this work contributes to a higher standard of methodological transparency in machine-learning-based neuroimaging research."),
];

// ---------------------------------------------------------------------------
// References (END) — peer-reviewed sources
// ---------------------------------------------------------------------------
const referencesBlock = [
  h1("References"),
  body("[1] GBD 2019 Dementia Forecasting Collaborators. Estimation of the global prevalence of dementia in 2019 and forecasted prevalence in 2050: an analysis for the Global Burden of Disease Study 2019. Lancet Public Health. 2022;7(2):e105–e125."),
  body("[2] Petersen RC. Mild cognitive impairment as a diagnostic entity. Journal of Internal Medicine. 2004;256(3):183–194."),
  body("[3] Albert MS, DeKosky ST, Dickson D, et al. The diagnosis of mild cognitive impairment due to Alzheimer’s disease: recommendations from the National Institute on Aging-Alzheimer’s Association workgroups. Alzheimer’s & Dementia. 2011;7(3):270–279."),
  body("[4] Mitchell AJ, Shiri-Feshki M. Rate of progression of mild cognitive impairment to dementia: meta-analysis of 41 robust inception cohort studies. Acta Psychiatrica Scandinavica. 2009;119(4):252–265."),
  body("[5] Petersen RC, Aisen PS, Beckett LA, et al. Alzheimer’s Disease Neuroimaging Initiative (ADNI): clinical characterization. Neurology. 2010;74(3):201–209."),
  body("[6] Ansart M, Epelbaum S, Bassignana G, et al. Predicting the progression of mild cognitive impairment using machine learning: a systematic, quantitative and critical review. Medical Image Analysis. 2021;67:101848."),
  body("[7] Moradi E, Pepe A, Gaser C, Huttunen H, Tohka J. Machine learning framework for early MRI-based Alzheimer’s conversion prediction in MCI subjects. NeuroImage. 2015;104:398–412."),
  body("[8] Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804."),
  body("[9] Liu CC, Kanekiyo T, Xu H, Bu G. Apolipoprotein E and Alzheimer disease: risk, mechanisms and therapy. Nature Reviews Neurology. 2013;9(2):106–118."),
  body("[10] Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016:785–794."),
  body("[11] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems. 2017;30:4765–4774."),
  body("[12] Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making. 2006;26(6):565–574."),
];

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------
const numberingConfig = ["contributions", "criteria", "shap"].map((ref) => ({
  reference: ref,
  levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 }, spacing: { after: 80, line: 276 } } } }],
}));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Times New Roman", color: "000000" },
        paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman", color: "000000" },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: numberingConfig },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Times New Roman" })] })] }) },
    children: [ ...titleBlock, ...abstractBlock, ...introBlock, ...methodsBlock, ...resultsBlock, ...discussionBlock, ...referencesBlock ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(DIR, "ADNI_paper_FINAL.docx"), buffer);
  console.log("Written: ADNI_paper_FINAL.docx (" + buffer.length + " bytes)");
});
