# Auspify Technologies - Data Science Internship Portfolio
**Candidate Name**: Redam Jaswanth  
**Position**: Data Science Intern  
**Internship Duration**: 11/10/2026 to 11/11/2026  
**Host Organization**: [Auspify Technologies](https://www.auspify.com) (`support@auspify.com`)  
**Task Submission Link**: [Auspify Task Submission Portal](https://docs.google.com/forms/d/e/1FAIpQLSeJLbjJ2H-neq-GHe_RPNkDXYEZ839MaO3TBIesZT8VSrbieQ/viewform?usp=dialog)  

---

## 📌 Project Overview
This repository contains the complete implementation of **all 6 practical Data Science tasks** assigned in the Auspify Technologies 4-Week Practical Learning Program. Every task has been designed, engineered, evaluated, and documented following professional industry standards.

The dataset analyzed is the official Netflix titles dataset sourced directly from Auspify Technologies.

---

## 📂 Directory Structure

```
C:\Auspify_Data_Science_Internship\
│
├── README.md                          # Project documentation and submission summary
├── requirements.txt                   # Project Python dependencies
├── run_all_tasks.py                   # Master automated runner for Tasks 1 - 6
├── app.py                             # Interactive Streamlit Web Dashboard (Task 6)
│
├── data\                              # Dataset repository
│   ├── Dataset.csv                    # Original raw dataset from Auspify (8,790 records)
│   └── cleaned_netflix_dataset.csv    # Preprocessed and engineered dataset (8,787 records)
│
├── outputs\                           # Generated artifacts
│   ├── figures\                       # High-resolution visualization charts (PNG)
│   │   ├── eda_1_content_distribution.png
│   │   ├── eda_2_top_countries.png
│   │   ├── eda_3_top_genres.png
│   │   ├── eda_4_release_year_trends.png
│   │   ├── eda_5_rating_distribution.png
│   │   ├── eda_6_movie_durations.png
│   │   ├── rec_similarity_distribution.png
│   │   ├── trend_forecast_total_additions.png
│   │   ├── trend_forecast_movies_vs_tv.png
│   │   ├── trend_tv_ratio_evolution.png
│   │   ├── classification_model_comparison.png
│   │   ├── classification_roc_curves.png
│   │   └── classification_confusion_matrix.png
│   └── models\                        # Serialized machine learning models
│       ├── tfidf_vectorizer.pkl       # NLP TF-IDF vectorizer artifact
│       ├── rec_catalog.parquet        # Recommendation search engine index
│       └── best_classifier_model.pkl  # Trained Random Forest classifier pipeline
│
├── notebooks\                         # Modular source code scripts
│   ├── task1_data_cleaning.py
│   ├── task2_exploratory_data_analysis.py
│   ├── task3_recommendation_system.py
│   ├── task4_trend_prediction_analysis.py
│   ├── task5_classification_model.py
│   └── task6_dashboard.py
│
└── reports\                           # Detailed technical documentation reports
    ├── Task1_Data_Cleaning_Report.md
    ├── Task2_EDA_Report.md
    ├── Task3_Recommendation_Report.md
    ├── Task4_Trend_Forecasting_Report.md
    ├── Task5_Classification_Report.md
    └── Task6_Business_Insights_Report.md
```

---

## 🚀 Tasks Breakdown & Deliverables

### ✅ Task 1: Data Cleaning & Preprocessing (Easy)
- **Workflow**: Checked missing values, removed 3 duplicate records, converted `date_added` into standard datetime, extracted `year_added`, `month_added`, `day_added`, segmented duration into numerical minutes/seasons, isolated primary genres and countries.
- **Output**: Cleaned dataset saved at `data/cleaned_netflix_dataset.csv` with 20 engineered columns.
- **Report**: `reports/Task1_Data_Cleaning_Report.md`

### ✅ Task 2: Exploratory Data Analysis (EDA) (Easy)
- **Workflow**: Conducted deep statistical profiling across catalog types (69.7% Movies vs 30.3% TV Shows), top 10 content-producing nations (US, India, UK, Canada), top 15 genres, historical release trends (1990–2021), and movie duration distributions (Mean runtime: 99.6 minutes).
- **Output**: 6 high-resolution publication charts generated in `outputs/figures/`.
- **Report**: `reports/Task2_EDA_Report.md`

### ✅ Task 3: Content-Based Recommendation System (Medium)
- **Workflow**: Built a metadata feature soup combining title, genres, director, country, and ratings. Computed TF-IDF vectorization with unigrams and bigrams and calculated cosine similarity.
- **Performance**: Achieved **94.07% primary genre consistency rate** across top recommendations with sub-second response times.
- **Artifacts**: Serialized `tfidf_vectorizer.pkl` and `rec_similarity_distribution.png`.
- **Report**: `reports/Task3_Recommendation_Report.md`

### ✅ Task 4: Trend Prediction Analysis (Medium)
- **Workflow**: Aggregated annual content additions from 2014 to 2021. Fitted regularized Polynomial Ridge regression ($R^2 = 0.7859$) to project 5-year forward additions (2022–2026) with 95% confidence intervals.
- **Key Insight**: Content acquisitions transition from parabolic catalog building to steady-state high-engagement episodic TV additions.
- **Artifacts**: 3 time-series forecast visualizations in `outputs/figures/`.
- **Report**: `reports/Task4_Trend_Forecasting_Report.md`

### ✅ Task 5: Machine Learning Classification Model (Advanced)
- **Workflow**: Built automated pipeline to classify content maturity rating (`Mature: TV-MA/R` vs `General/Teens`). Evaluated Logistic Regression, Random Forest, and Gradient Boosting under stratified 80/20 train/test split.
- **Champion Model**: Random Forest Classifier:
  - **ROC-AUC**: `0.7754`
  - **F1-Score**: `0.7096`
  - **Recall on Mature Content**: `82.89%`
- **Artifacts**: `best_classifier_model.pkl`, ROC curve plot, and confusion matrix heatmap.
- **Report**: `reports/Task5_Classification_Report.md`

### ✅ Task 6: Data Science Business Insights Dashboard (Advanced)
- **Workflow**: Developed a full-stack interactive Streamlit web dashboard (`app.py`) featuring 6 specialized portals:
  1. Executive KPI Overview
  2. Interactive Exploratory Analytics
  3. Semantic Content Recommender
  4. Trend Forecasting Studio
  5. Live ML Rating Predictor Form
  6. Strategic Business Roadmap
- **Report**: `reports/Task6_Business_Insights_Report.md`

---

## 💻 Installation & Quickstart

### 1. Install Dependencies
```bash
cd C:\Auspify_Data_Science_Internship
pip install -r requirements.txt
```

### 2. Run All Tasks Pipeline
To execute all tasks end-to-end and reproduce all figures, models, and reports:
```bash
python C:\Auspify_Data_Science_Internship\run_all_tasks.py
```

### 3. Launch Interactive Dashboard
```bash
streamlit run C:\Auspify_Data_Science_Internship\app.py
```

---

## 🎯 Auspify Evaluation Criteria Alignment
| Evaluation Parameter | Project Implementation Status |
|---|---|
| **Task Completion** | ✅ 6 out of 6 tasks fully implemented (exceeding minimum requirement of 4) |
| **Code Quality** | ✅ Modular, clean, type-safe, PEP-8 compliant Python scripts |
| **User Interface Design** | ✅ Sleek Netflix-themed interactive Streamlit Web Dashboard |
| **Creativity & Implementation**| ✅ Combined NLP similarity, polynomial regression forecasting, and ensemble ML |
| **Professional Presentation**| ✅ Formal Markdown technical reports for every single task |
| **Project Functionality** | ✅ End-to-end verified with serialized models and reproducible pipeline |

---
*Created for submission to Auspify Technologies Data Science Internship Program.*
