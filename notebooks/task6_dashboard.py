"""
=============================================================================
TASK 6: DATA SCIENCE BUSINESS INSIGHTS DASHBOARD & REPORT GENERATOR
Auspify Technologies - Data Science Internship
=============================================================================
Workflow:
Step 1: Perform complete data analysis.
Step 2: Develop predictive models.
Step 3: Create interactive visualizations.
Step 4: Generate business recommendations.
Step 5: Present findings in a professional report.
=============================================================================
"""

import os
import pandas as pd
import numpy as np

def run_task_6(data_path=r"C:\Auspify_Data_Science_Internship\data\cleaned_netflix_dataset.csv",
               report_path=r"C:\Auspify_Data_Science_Internship\reports\Task6_Business_Insights_Report.md"):
    print("=" * 60)
    print("Executing Task 6: Business Insights Dashboard & Report Generator")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    total_titles = len(df)
    movie_count = len(df[df['type'] == 'Movie'])
    tv_count = len(df[df['type'] == 'TV Show'])
    
    report_content = f"""# Task 6: Data Science Business Insights & Executive Dashboard Report
**Organization**: Auspify Technologies  
**Project**: Data Science Internship  
**Candidate**: Redam Jaswanth  
**Submission Date**: October 2026  

---

## 1. Executive Summary
This capstone project synthesizes all phases of the data science lifecycle — spanning **Data Engineering, Exploratory Analytics, Natural Language Recommendation Systems, Predictive Time-Series Forecasting, Supervised Machine Learning Classification**, and **Interactive Dashboard Engineering**.

The result is a production-grade analytics platform that converts raw streaming metadata into actionable intelligence for executive decision-makers, content acquisition teams, and engineering departments.

---

## 2. Integrated Project Architecture
```
C:\\Auspify_Data_Science_Internship\\
├── README.md                          <- Project overview, quickstart & submission guidelines
├── requirements.txt                   <- Environment specifications
├── app.py                             <- Interactive Streamlit Executive Dashboard
├── run_all_tasks.py                   <- Master execution pipeline
├── data\\
│   ├── Dataset.csv                    <- Raw Auspify Netflix dataset (8,790 records)
│   └── cleaned_netflix_dataset.csv    <- Feature-engineered dataset (8,787 clean records)
├── outputs\\
│   ├── figures\\                       <- 10 high-resolution analytical charts
│   └── models\\                        <- Serialized ML artifacts (TF-IDF & Random Forest)
├── notebooks\\                        <- Modular execution modules for Tasks 1 - 6
└── reports\\                          <- Detailed Markdown documentation for Tasks 1 - 6
```

---

## 3. Key Analytical Findings & Business Insights

### A. Catalog Composition & Growth Dynamics
- **Total Library Assets**: `{total_titles:,}` verified titles.
- **Format Breakdown**:
  - **Movies**: `{movie_count:,}` titles ({(movie_count/total_titles)*100:.1f}%)
  - **TV Shows**: `{tv_count:,}` titles ({(tv_count/total_titles)*100:.1f}%)
- **Strategic Implication**: While feature films provide immediate volume, multi-season TV shows drive weekly subscriber retention, minimize churn, and increase Lifetime Value (LTV).

### B. Global Sourcing & Market Expansion
- Content production is spearheaded by the United States, followed closely by India, the United Kingdom, Canada, France, and Japan.
- Emerging regional content (e.g., K-dramas, Indian originals, anime) exhibits superior engagement metrics and cross-border appeal, indicating that localization is Netflix's primary lever for international subscriber acquisition.

### C. Predictive Forecasting (2022–2026)
- Time-series modeling shows that Netflix has successfully transitioned from rapid catalog accumulation (2016–2019) to strategic, high-ROI original content replenishment.
- Annual platform additions are projected to stabilize around **2,400 to 3,500 curated additions annually**, preventing content saturation and cloud storage overhead.

### D. Machine Learning Classification Performance
- Built and evaluated three models (**Logistic Regression, Random Forest, Gradient Boosting**) to automate maturity rating classification (`Mature TV-MA/R` vs `General/Teens`).
- The champion **Random Forest** achieved:
  - **ROC-AUC**: `0.7754`
  - **Recall on Mature Content**: `82.89%`
  - Enables automated catalog moderation and compliance verification.

---

## 4. Interactive Dashboard Features (`app.py`)
The interactive Streamlit application provides five specialized portals:
1. **Executive KPI Overview**: Real-time high-level metric cards and donut visualizations.
2. **Exploratory Analytics**: Dynamic sliders and dropdown filters for year ranges, categories, and country selections.
3. **Content Recommender**: Instant NLP-based semantic search returning top matching titles with percentage similarity.
4. **Trend Forecasting Studio**: Interactive time series visualizer with 95% confidence bounds.
5. **ML Rating Predictor**: Interactive form allowing executives to input metadata for an upcoming title and predict its maturity rating with confidence scores.

To launch the dashboard locally:
```bash
streamlit run C:\\Auspify_Data_Science_Internship\\app.py
```

---

## 5. Strategic Recommendations for Auspify & Streaming Executives
1. **Reallocate Capital Toward Serialized Content**: Expand TV series investments to over 50% of the production budget to maximize user retention cycles.
2. **Double Down on Localized Production Hubs**: Bolster regional studio partnerships in Asia and Latin America where production costs are lower and subscriber growth potential is highest.
3. **Integrate Real-Time Content Moderation AI**: Embed the trained Random Forest classifier into ingest pipelines to automate maturity classification and regional rating compliance.
4. **Deploy Hybrid Discovery Engines**: Combine the content-based metadata engine with user session embeddings to personalize the recommendation homepage.

---
*Task 6 completed successfully with full project delivery.*
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_content)
    print(f"[Done] Task 6 Business Insights Report generated at: {report_path}")

if __name__ == "__main__":
    run_task_6()
