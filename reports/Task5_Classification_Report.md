# Task 5: Machine Learning Classification Model Report
**Organization**: Auspify Technologies  
**Project**: Data Science Internship  
**Prepared For**: Evaluation & Project Verification  

---

## 1. Executive Summary
This project develops an automated supervised Machine Learning classification pipeline designed to categorize Netflix content into its appropriate **Audience Maturity Profile** (`Mature: TV-MA/R/NC-17` vs `General/Teens: TV-14/PG/PG-13/G`). This classification enables automated catalog tagging, parental controls enforcement, and regional broadcast compliance.

---

## 2. Experimental Setup
- **Sample Distribution**:
  - Total Samples: `8,787`
  - Training Set (80% Stratified): `7,029` records
  - Testing Set (20% Stratified): `1,758` records
- **Engineered Features**:
  - Numeric: `release_year`, `content_age_years`, `has_director`
  - Categorical (One-Hot Encoded): `type` (Movie/TV Show), `country_feat` (Top 10 producing countries), `genre_feat` (Top 15 primary genres)
- **Class Imbalance Strategy**: Balanced class weighting applied across estimators.

---

## 3. Comparative Model Evaluation Leaderboard
Three distinct algorithms were evaluated across five standardized statistical metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.6843 | 0.6203 | 0.8289 | 0.7096 | 0.7754 |
| Logistic Regression | 0.6843 | 0.6360 | 0.7518 | 0.6891 | 0.7650 |
| Gradient Boosting | 0.6860 | 0.6430 | 0.7311 | 0.6842 | 0.7670 |

---

## 4. Champion Model Selection: Random Forest
The **Random Forest** model demonstrated the strongest overall discriminative capability, achieving:
- **Accuracy**: `68.43%`
- **F1-Score**: `0.7096`
- **ROC-AUC**: `0.7754`

### Performance Visualizations:
- `classification_model_comparison.png`: Comprehensive metric bar chart across all three architectures.
- `classification_roc_curves.png`: Discrimination capability curves across sensitivity thresholds.
- `classification_confusion_matrix.png`: True positive, false positive, true negative, and false negative counts.

---

## 5. Artifact Delivery
- Serialized Model Pipeline: `outputs/models/best_classifier_model.pkl`
- Deployment Readiness: Fully integrated into the Streamlit Web Application for real-time single-record inference.

---
*Task 5 completed successfully.*
