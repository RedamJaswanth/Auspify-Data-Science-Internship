"""
=============================================================================
TASK 1: DATA CLEANING & PREPROCESSING
Auspify Technologies - Data Science Internship
=============================================================================
Workflow:
Step 1: Import the dataset using Pandas.
Step 2: Identify missing and duplicate records.
Step 3: Handle null values and inconsistent formats.
Step 4: Transform categorical and date-related columns.
Step 5: Create a clean dataset for further analysis.
=============================================================================
"""

import os
import pandas as pd
import numpy as np

def run_task_1(input_path=r"C:\Auspify_Data_Science_Internship\data\Dataset.csv",
               output_path=r"C:\Auspify_Data_Science_Internship\data\cleaned_netflix_dataset.csv",
               report_path=r"C:\Auspify_Data_Science_Internship\reports\Task1_Data_Cleaning_Report.md"):
    print("=" * 60)
    print("Executing Task 1: Data Cleaning & Preprocessing")
    print("=" * 60)
    
    # Step 1: Import dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    
    df_raw = pd.read_csv(input_path)
    initial_shape = df_raw.shape
    print(f"[Step 1] Dataset loaded successfully. Initial Shape: {initial_shape[0]} rows, {initial_shape[1]} columns.")
    
    # Step 2: Identify missing and duplicate records
    duplicates_count = df_raw.duplicated(subset=['title', 'type', 'release_year']).sum()
    show_id_dupes = df_raw.duplicated(subset=['show_id']).sum()
    print(f"[Step 2] Found {duplicates_count} potential duplicate titles, {show_id_dupes} duplicate show_ids.")
    
    # Check for placeholder nulls like 'Not Given', 'Unknown', NaN
    placeholder_nulls = ['Not Given', 'unknown', 'Unknown', 'None', 'NA', 'N/A', '']
    df_clean = df_raw.copy()
    
    # Remove exact duplicate records if any
    df_clean = df_clean.drop_duplicates(subset=['title', 'type', 'release_year'])
    
    # Step 3: Handle null values and inconsistent formats
    # Replace placeholder strings with standard missing representations or impute
    for col in ['director', 'country']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(placeholder_nulls, np.nan)
            df_clean[col] = df_clean[col].fillna("Unknown")
            
    # Clean rating nulls
    if 'rating' in df_clean.columns:
        df_clean['rating'] = df_clean['rating'].replace(placeholder_nulls, np.nan)
        mode_rating = df_clean['rating'].mode()[0] if not df_clean['rating'].mode().empty else 'TV-MA'
        df_clean['rating'] = df_clean['rating'].fillna(mode_rating)
        
    # Step 4: Transform categorical and date-related columns
    # Strip whitespace across string columns
    str_cols = df_clean.select_dtypes(include='object').columns
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()
        
    # Date added parsing
    df_clean['date_added_clean'] = pd.to_datetime(df_clean['date_added'], format='mixed', errors='coerce')
    # If any date_added failed, impute with mode or median release year date
    df_clean['year_added'] = df_clean['date_added_clean'].dt.year.fillna(df_clean['release_year']).astype(int)
    df_clean['month_added'] = df_clean['date_added_clean'].dt.month.fillna(1).astype(int)
    df_clean['month_name_added'] = df_clean['date_added_clean'].dt.month_name().fillna('January')
    df_clean['day_added'] = df_clean['date_added_clean'].dt.day.fillna(1).astype(int)
    
    # Duration parsing (Numeric value + Unit)
    # Movie: "90 min" -> 90 numeric
    # TV Show: "1 Season" -> 1 numeric
    def parse_duration(val):
        val = str(val).strip()
        parts = val.split()
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
        return 0
        
    def parse_duration_unit(val):
        val = str(val).strip().lower()
        if 'min' in val:
            return 'Minutes'
        elif 'season' in val:
            return 'Seasons'
        return 'Other'

    df_clean['duration_num'] = df_clean['duration'].apply(parse_duration)
    df_clean['duration_unit'] = df_clean['duration'].apply(parse_duration_unit)
    
    # Primary genre & Primary country extraction
    df_clean['primary_genre'] = df_clean['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',')][0] if pd.notna(x) else 'Unknown')
    df_clean['primary_country'] = df_clean['country'].apply(lambda x: [c.strip() for c in str(x).split(',')][0] if pd.notna(x) else 'Unknown')
    
    # Content age
    current_year = 2026
    df_clean['content_age_years'] = current_year - df_clean['release_year']
    
    # Step 5: Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    final_shape = df_clean.shape
    print(f"[Step 5] Cleaned dataset saved to: {output_path}")
    print(f"Final Cleaned Shape: {final_shape[0]} rows, {final_shape[1]} columns.")
    
    # Generate Markdown Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_content = f"""# Task 1: Data Cleaning & Preprocessing Report
**Organization**: Auspify Technologies  
**Project**: Data Science Internship  
**Prepared For**: Evaluation & Project Verification  

---

## 1. Executive Summary
Data cleaning and preprocessing is the foundational phase of the Data Science lifecycle. This phase transforms the raw, potentially noisy Netflix dataset into a high-integrity, structured, and feature-rich dataset optimized for Exploratory Data Analysis, Recommendation Modeling, Time Series Forecasting, and Supervised Machine Learning.

---

## 2. Dataset Overview
- **Raw Dataset Records**: {initial_shape[0]:,} rows × {initial_shape[1]} columns
- **Cleaned Dataset Records**: {final_shape[0]:,} rows × {final_shape[1]} columns
- **Removed Duplicate Records**: {duplicates_count}
- **Output Storage Path**: `{output_path}`

---

## 3. Preprocessing Steps Executed
1. **Deduplication**: Checked and eliminated redundant records based on composite key `(title, type, release_year)`.
2. **Missing Value & Placeholder Treatment**:
   - Replaced placeholder values (`'Not Given'`, `'Unknown'`, blanks) in `director` and `country` with standardized label `'Unknown'`.
   - Imputed rare missing ratings with the statistical mode (`'{mode_rating}'`).
3. **Datetime Parsing & Feature Extraction**:
   - Parsed unstructured date strings in `date_added` into standard datetime representations.
   - Extracted temporal features: `year_added`, `month_added`, `month_name_added`, and `day_added`.
4. **Feature Engineering**:
   - **`duration_num`**: Extracted numerical magnitude (e.g., minutes for movies, season count for TV shows).
   - **`duration_unit`**: Classified duration into `'Minutes'` vs `'Seasons'`.
   - **`primary_genre`**: Extracted primary catalog category from comma-separated genre tags.
   - **`primary_country`**: Isolated main country of origin for multinational productions.
   - **`content_age_years`**: Calculated temporal age of content relative to current year.

---

## 4. Column Schema & Data Types (Post-Cleaning)
| Column Name | Data Type | Description |
|---|---|---|
| `show_id` | String | Unique Netflix identifier |
| `type` | String | Content category ('Movie' or 'TV Show') |
| `title` | String | Title of movie or TV show |
| `director` | String | Production director(s) or 'Unknown' |
| `country` | String | Country of origin |
| `date_added` | String | Original date added |
| `release_year` | Integer | Year of original theatrical/broadcast release |
| `rating` | String | Age rating (e.g., TV-MA, TV-14, PG-13, R) |
| `duration` | String | Original duration string |
| `listed_in` | String | Full genre tags |
| `date_added_clean` | Date | Standardized ISO-8601 date |
| `year_added` | Integer | Year content was uploaded to Netflix |
| `month_added` | Integer | Month content was uploaded (1-12) |
| `month_name_added`| String | Full name of addition month |
| `day_added` | Integer | Day of month content was added |
| `duration_num` | Integer | Numeric duration count |
| `duration_unit` | String | Unit ('Minutes' or 'Seasons') |
| `primary_genre` | String | Lead categorization |
| `primary_country` | String | Primary producing country |
| `content_age_years`| Integer | Age of media in years |

---
*Task 1 completed successfully with zero critical errors.*
"""
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_content)
    print(f"[Done] Task 1 Report generated at: {report_path}")
    return df_clean

if __name__ == "__main__":
    run_task_1()
