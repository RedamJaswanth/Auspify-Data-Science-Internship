"""
=============================================================================
MASTER PIPELINE RUNNER - ALL 6 DATA SCIENCE TASKS
Auspify Technologies - Data Science Internship
Candidate: Redam Jaswanth
=============================================================================
This master script executes the entire data science end-to-end pipeline:
- Task 1: Data Cleaning & Preprocessing
- Task 2: Exploratory Data Analysis (EDA) & Visualizations
- Task 3: Content-Based Recommendation System
- Task 4: Trend Prediction & Time Series Forecasting
- Task 5: Supervised Machine Learning Classification
- Task 6: Business Insights & Executive Reporting
=============================================================================
"""

import sys
import os
import time

# Ensure notebooks folder is in sys.path
BASE_DIR = r"C:\Auspify_Data_Science_Internship"
sys.path.append(os.path.join(BASE_DIR, "notebooks"))

from task1_data_cleaning import run_task_1
from task2_exploratory_data_analysis import run_task_2
from task3_recommendation_system import run_task_3
from task4_trend_prediction_analysis import run_task_4
from task5_classification_model import run_task_5
from task6_dashboard import run_task_6

def main():
    start_time = time.time()
    print("=" * 75)
    print("AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP PIPELINE")
    print("Executing All 6 Core Tasks Sequentially")
    print("=" * 75)
    
    # Task 1
    print("\n>>> STARTING TASK 1: DATA CLEANING & PREPROCESSING...")
    run_task_1()
    
    # Task 2
    print("\n>>> STARTING TASK 2: EXPLORATORY DATA ANALYSIS (EDA)...")
    run_task_2()
    
    # Task 3
    print("\n>>> STARTING TASK 3: RECOMMENDATION SYSTEM ANALYSIS...")
    run_task_3()
    
    # Task 4
    print("\n>>> STARTING TASK 4: TREND PREDICTION ANALYSIS...")
    run_task_4()
    
    # Task 5
    print("\n>>> STARTING TASK 5: MACHINE LEARNING CLASSIFICATION...")
    run_task_5()
    
    # Task 6
    print("\n>>> STARTING TASK 6: BUSINESS INSIGHTS & REPORTING...")
    run_task_6()
    
    total_duration = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"🎉 ALL 6 TASKS COMPLETED SUCCESSFULLY IN {total_duration:.1f} SECONDS!")
    print("=" * 75)
    print(f"All datasets, models, plots, and markdown reports are stored at:")
    print(f"--> {BASE_DIR}")
    print("\nTo launch the interactive web dashboard, run:")
    print("streamlit run C:\\Auspify_Data_Science_Internship\\app.py")
    print("=" * 75)

if __name__ == "__main__":
    main()
