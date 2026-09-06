# Maternal Health Risk Stratification - Interpretable Machine Learning with Python-lianafarsi

## 1. Project Overview & Clinical Problem
This project explores patterns associated with maternal health risk levels using a small tabular dataset[cite: 1]. The goal is to build a simple baseline classifier separating patients into Low, Mid, and High-Risk groups, focusing on careful health-data analysis, appropriate evaluation, and cautious interpretation[cite: 1].

## 2. Dataset Information
- **Source:** UCI Machine Learning Repository (Maternal Health Risk)[cite: 1]
- **Link:** https://archive.ics.uci.edu/dataset/863/pregnant+health+risk[cite: 1]
- **DOI:** 10.24432/C5DP5D[cite: 1]
- **License:** CC BY 4.0[cite: 1]
- **Context:** Collected through an IoT-based risk-monitoring system in rural Bangladesh[cite: 1]. Contains 1,013 instances and 6 input features (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate) plus a categorical `RiskLevel`[cite: 1].

## 3. Data Quality & EDA Findings
- **Quality:** No missing values or severe implausible outliers were found[cite: 1].
- **EDA:** High-Risk patients tend to cluster at significantly higher levels of Blood Sugar (BS) and Systolic/Diastolic Blood Pressure compared to Low and Mid-Risk groups[cite: 1].

## 4. Modeling & Evaluation
- **Model:** A shallow **Decision Tree** was used as a simple baseline classifier to maintain interpretability[cite: 1]. Complex ensembles were avoided[cite: 1].
- **Evaluation:** Overall accuracy was not the primary metric[cite: 1]. The model was evaluated using a Confusion Matrix, Recall, and F1-score[cite: 1]. Maximizing Recall for the **High Risk** class is critical, as missing a high-risk case (False Negative) has severe clinical consequences[cite: 1].

## 5. Error Analysis & Feature Interpretation
- **Errors:** Misclassifications mainly occurred between the overlapping boundaries of 'Low' and 'Mid' risk classes[cite: 1].
- **Feature Importance:** Blood Sugar (BS) and Systolic Blood Pressure (SystolicBP) were the strongest contributors driving the model's predictions[cite: 1].

## 6. Clinical Cautions & Limitations
**This is an educational risk-stratification project, not a diagnostic tool[cite: 1].**
- **Association vs. Causation:** The high feature importance of blood sugar does not prove clinical causation[cite: 1].
- **Dataset Limitations:** The analysis relies on a small observational dataset strictly from rural Bangladesh, lacking external validation across other geographies[cite: 1]. 
- **Real-world Use:** Clinical application would require massive, diverse datasets and professional medical validation[cite: 1].

## 7. Repository Structure
- `README.md`: Project documentation and findings[cite: 1]
- `requirements.txt`: Python dependencies[cite: 1]
- `notebooks/analysis.ipynb`: Main analysis and modeling code[cite: 1]
- `figures/`: Exported visualizations (Risk Distribution, Blood Sugar Boxplot, BP Scatter, Confusion Matrix, Feature Importance)[cite: 1]

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `notebooks/analysis.ipynb` cell by cell. Ensure the dataset is loaded properly.
