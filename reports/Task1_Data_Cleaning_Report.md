# Task 1: Data Cleaning & Preprocessing Report
**Organization**: Auspify Technologies  
**Project**: Data Science Internship  
**Prepared For**: Evaluation & Project Verification  

---

## 1. Executive Summary
Data cleaning and preprocessing is the foundational phase of the Data Science lifecycle. This phase transforms the raw, potentially noisy Netflix dataset into a high-integrity, structured, and feature-rich dataset optimized for Exploratory Data Analysis, Recommendation Modeling, Time Series Forecasting, and Supervised Machine Learning.

---

## 2. Dataset Overview
- **Raw Dataset Records**: 8,790 rows × 10 columns
- **Cleaned Dataset Records**: 8,787 rows × 20 columns
- **Removed Duplicate Records**: 3
- **Output Storage Path**: `C:\Auspify_Data_Science_Internship\data\cleaned_netflix_dataset.csv`

---

## 3. Preprocessing Steps Executed
1. **Deduplication**: Checked and eliminated redundant records based on composite key `(title, type, release_year)`.
2. **Missing Value & Placeholder Treatment**:
   - Replaced placeholder values (`'Not Given'`, `'Unknown'`, blanks) in `director` and `country` with standardized label `'Unknown'`.
   - Imputed rare missing ratings with the statistical mode (`'TV-MA'`).
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
