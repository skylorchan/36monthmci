# 36monthmci
🧠 Explainable Multimodal Deep Learning for Early Alzheimer’s Prediction
Overview

This project develops an explainable multimodal deep learning framework to predict the conversion from Mild Cognitive Impairment (MCI) to Alzheimer’s Disease (AD) within a 3-year horizon.

Unlike traditional models that rely on single-modality data, this approach integrates:

Clinical and cognitive assessments
MRI-derived brain region features
PET imaging biomarkers
Genetic information (e.g., APOE status)

The model focuses not only on prediction accuracy, but also on interpretability, identifying which features and timepoints drive disease progression.

Problem Statement

Early detection of Alzheimer’s disease remains a major challenge in neuroscience and clinical practice.

Patients with MCI may or may not convert to AD. Predicting this conversion early:

Enables earlier intervention
Improves patient outcomes
Supports clinical trial recruitment

However, current approaches:

Ignore longitudinal structure
Fail to integrate multiple modalities
Lack interpretability

This project addresses all three.

Key Features
📊 Multimodal Learning: Combines MRI, PET, cognitive scores, and genetics
⏱ Longitudinal Modeling: Uses time-series patient data (irregular visits)
⚙️ Transformer-Based Architecture for temporal modeling
📈 Survival Analysis Framework (predicts risk over time, not just binary outcome)
🔍 Explainability Tools:
Temporal SHAP values
Feature importance by modality
Brain-region attribution maps
Dataset

This project uses publicly available datasets:

ADNI (Alzheimer’s Disease Neuroimaging Initiative)
OASIS-3

Data includes:

Cognitive tests (MMSE, CDR, ADAS-Cog, FAQ)
MRI (FreeSurfer ROI volumes & cortical thickness)
PET (amyloid + FDG SUVR values)
Demographics and APOE genotype

Note: Data access requires registration and agreement with dataset usage policies.

Methodology
1. Cohort Construction
Identify baseline MCI patients
Label:
Converters: MCI → AD within 36 months
Non-converters: No conversion within timeframe
Handle censoring using survival modeling
2. Preprocessing
Harmonization across sites (e.g., ComBat)
Missing data handling with masking
Temporal alignment using visit timestamps
3. Model Architecture
Modality-specific encoders:
Clinical branch
Imaging branch
Genetic/static features
Fusion via cross-modal attention
Temporal modeling via:
Transformer / GRU-based sequence encoder
Output:
Discrete-time hazard function
Binary conversion probability (auxiliary task)
4. Training
Loss:
Survival loss (negative log-likelihood)
Binary cross-entropy (conversion task)
Regularization:
Dropout
Early stopping
Evaluation Metrics
C-index (survival performance)
Time-dependent AUC (12, 24, 36 months)
Integrated Brier Score
AUC / AUPRC for classification
Calibration analysis
Explainability

To ensure clinical relevance, the model includes:

Temporal SHAP → which features matter at which time
Modality importance weights → which data source drives predictions
Brain region mapping → highlights key neuroanatomical changes
Project Structure
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── cohort_builder.ipynb
│   ├── preprocessing.ipynb
├── models/
│   ├── multimodal_transformer.py
├── training/
│   ├── train.py
│   ├── evaluate.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
├── results/
│   ├── figures/
│   ├── logs/
└── README.md
Installation
git clone https://github.com/yourusername/alzheimers-multimodal-prediction.git
cd alzheimers-multimodal-prediction

pip install -r requirements.txt
Usage
1. Build Cohort
python notebooks/cohort_builder.ipynb
2. Train Model
python training/train.py
3. Evaluate
python training/evaluate.py
Results (Example)
C-index: ~0.78
AUC (36-month conversion): ~0.82
Strong predictive features:
Hippocampal atrophy
CDR-SB progression
Amyloid PET SUVR
Future Work
Incorporate raw imaging (CNN-based feature extraction)
Add diffusion MRI and tau PET
Improve handling of missing modalities
Deploy as a clinical decision-support tool
References
ADNI Database
OASIS-3 Dataset
Relevant deep learning + survival analysis literature
License

MIT License

Acknowledgments
Alzheimer’s Disease Neuroimaging Initiative (ADNI)
OASIS-3 contributors
Open-source ML community
