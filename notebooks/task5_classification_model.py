"""
=============================================================================
TASK 5: MACHINE LEARNING CLASSIFICATION MODEL
Auspify Technologies - Data Science Internship
=============================================================================
Workflow:
Step 1: Select and prepare features (Encoding, Imputation, Text TF-IDF / One-Hot).
Step 2: Split data into training and testing sets (Stratified 80/20).
Step 3: Train classification models (Logistic Regression, Random Forest, Gradient Boosting).
Step 4: Evaluate model performance (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
Step 5: Compare model accuracy and serialize best champion model.
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_task_5(data_path=r"C:\Auspify_Data_Science_Internship\data\cleaned_netflix_dataset.csv",
               models_dir=r"C:\Auspify_Data_Science_Internship\outputs\models",
               figures_dir=r"C:\Auspify_Data_Science_Internship\outputs\figures",
               report_path=r"C:\Auspify_Data_Science_Internship\reports\Task5_Classification_Report.md"):
    print("=" * 60)
    print("Executing Task 5: Machine Learning Classification Model")
    print("=" * 60)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Step 1: Select and prepare features
    # Task: Classify Content Maturity Profile (Mature Audiences TV-MA/R/NC-17 vs General/Teens PG-13/TV-14/PG/G)
    # This represents a core business problem: automated content rating moderation for streaming catalogs
    df = pd.read_csv(data_path)
    
    # Target definition: 1 if Mature (TV-MA, R, NC-17), 0 if General/Teens
    mature_ratings = ['TV-MA', 'R', 'NC-17', 'NR', 'UR']
    df['is_mature'] = df['rating'].apply(lambda r: 1 if str(r).strip() in mature_ratings else 0)
    
    # Feature Engineering
    df['has_director'] = (df['director'] != 'Unknown').astype(int)
    
    # Filter top 10 countries and genres to prevent high dimensionality sparsity
    top_countries = set(df['primary_country'].value_counts().head(10).index)
    df['country_feat'] = df['primary_country'].apply(lambda c: c if c in top_countries else 'Other')
    
    top_genres = set(df['primary_genre'].value_counts().head(15).index)
    df['genre_feat'] = df['primary_genre'].apply(lambda g: g if g in top_genres else 'Other')
    
    features_num = ['release_year', 'content_age_years', 'has_director']
    features_cat = ['type', 'country_feat', 'genre_feat']
    
    X = df[features_num + features_cat]
    y = df['is_mature']
    
    print(f"[Step 1] Selected features: {features_num + features_cat}")
    print(f"Target distribution: Mature={y.sum()} ({y.mean()*100:.1f}%), Non-Mature={(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # Step 2: Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Step 2] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
        ]
    )
    
    # Step 3: Train Multiple Classification Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    roc_curves = {}
    conf_matrices = {}
    
    for name, clf in models.items():
        print(f"\n[Step 3] Training {name}...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        # Step 4: Evaluate Model Performance
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, roc_auc)
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    print("\n[Step 5] Model Comparison Leaderboard:")
    print(results_df.to_string(index=False))
    
    # Champion Model Selection
    champion_name = results_df.iloc[0]['Model']
    champion_pipe = trained_pipelines[champion_name]
    print(f"\nChampion Model: {champion_name}")
    
    # Serialize Champion Model
    model_save_path = os.path.join(models_dir, "best_classifier_model.pkl")
    joblib.dump(champion_pipe, model_save_path)
    print(f"Serialized champion model to: {model_save_path}")
    
    # Visualization 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    df_metrics_melt = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    sns.barplot(data=df_metrics_melt, x='variable', y='value', hue='Model', palette=['#E50914', '#221F1F', '#564d4d'])
    plt.title("Classification Performance Across Evaluated Models", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "classification_model_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # Visualization 2: ROC Curves
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, score) in roc_curves.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Chance (AUC = 0.50)')
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "classification_roc_curves.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    # Visualization 3: Confusion Matrix of Champion Model
    plt.figure(figsize=(7, 6))
    cm = conf_matrices[champion_name]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['General/Teens (0)', 'Mature (1)'],
                yticklabels=['General/Teens (0)', 'Mature (1)'])
    plt.title(f"Confusion Matrix: {champion_name}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "classification_confusion_matrix.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # Step 5: Generate Report
    def df_to_md(d):
        header = "| " + " | ".join(d.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
        rows = ["| " + " | ".join(f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val) for val in row) + " |" for row in d.values]
        return "\n".join([header, sep] + rows)

    report_content = f"""# Task 5: Machine Learning Classification Model Report
**Organization**: Auspify Technologies  
**Project**: Data Science Internship  
**Prepared For**: Evaluation & Project Verification  

---

## 1. Executive Summary
This project develops an automated supervised Machine Learning classification pipeline designed to categorize Netflix content into its appropriate **Audience Maturity Profile** (`Mature: TV-MA/R/NC-17` vs `General/Teens: TV-14/PG/PG-13/G`). This classification enables automated catalog tagging, parental controls enforcement, and regional broadcast compliance.

---

## 2. Experimental Setup
- **Sample Distribution**:
  - Total Samples: `{len(df):,}`
  - Training Set (80% Stratified): `{len(X_train):,}` records
  - Testing Set (20% Stratified): `{len(X_test):,}` records
- **Engineered Features**:
  - Numeric: `release_year`, `content_age_years`, `has_director`
  - Categorical (One-Hot Encoded): `type` (Movie/TV Show), `country_feat` (Top 10 producing countries), `genre_feat` (Top 15 primary genres)
- **Class Imbalance Strategy**: Balanced class weighting applied across estimators.

---

## 3. Comparative Model Evaluation Leaderboard
Three distinct algorithms were evaluated across five standardized statistical metrics:

{df_to_md(results_df)}

---

## 4. Champion Model Selection: {champion_name}
The **{champion_name}** model demonstrated the strongest overall discriminative capability, achieving:
- **Accuracy**: `{results_df.iloc[0]['Accuracy'] * 100:.2f}%`
- **F1-Score**: `{results_df.iloc[0]['F1-Score']:.4f}`
- **ROC-AUC**: `{results_df.iloc[0]['ROC-AUC']:.4f}`

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
"""
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_content)
    print(f"[Done] Task 5 Classification Report generated at: {report_path}")

if __name__ == "__main__":
    run_task_5()
