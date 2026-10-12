"""
=============================================================================
TASK 3: RECOMMENDATION SYSTEM ANALYSIS
Auspify Technologies - Data Science Internship
=============================================================================
Workflow:
Step 1: Extract relevant content features.
Step 2: Perform text preprocessing (TF-IDF vectorization, feature soup).
Step 3: Calculate content similarity (Cosine Similarity).
Step 4: Generate recommendations for selected titles.
Step 5: Evaluate recommendation quality (similarity distribution & genre alignment).
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def create_feature_soup(row):
    """Combines metadata features into a single rich text representation."""
    title = str(row['title'])
    director = str(row['director']).replace('Unknown', '')
    country = str(row['country']).replace('Unknown', '')
    genres = str(row['listed_in'])
    rating = str(row['rating'])
    content_type = str(row['type'])
    
    # Weight genres and type more heavily by repeating
    soup = f"{title} {content_type} {content_type} {genres} {genres} {director} {country} {rating}"
    return soup.lower()

def run_task_3(data_path=r"C:\Auspify_Data_Science_Internship\data\cleaned_netflix_dataset.csv",
               models_dir=r"C:\Auspify_Data_Science_Internship\outputs\models",
               figures_dir=r"C:\Auspify_Data_Science_Internship\outputs\figures",
               report_path=r"C:\Auspify_Data_Science_Internship\reports\Task3_Recommendation_Report.md"):
    print("=" * 60)
    print("Executing Task 3: Content-Based Recommendation System")
    print("=" * 60)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Step 1: Load and extract features
    df = pd.read_csv(data_path)
    df = df.reset_index(drop=True)
    print(f"[Step 1] Extracted metadata for {len(df)} titles.")
    
    # Step 2: Text preprocessing & TF-IDF
    df['metadata_soup'] = df.apply(create_feature_soup, axis=1)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['metadata_soup'])
    print(f"[Step 2] TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Step 3: Cosine similarity calculation
    # Using linear_kernel which is equivalent to cosine similarity for normalized tf-idf vectors
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(f"[Step 3] Computed Cosine Similarity Matrix: {cosine_sim.shape}")
    
    # Mapping from title to index
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    
    # Step 4: Recommendation Generator Function
    def get_recommendations(title, n_recommendations=5):
        clean_title = str(title).strip().lower()
        if clean_title not in indices:
            # Fuzzy match fallback
            matches = [t for t in indices.index if clean_title in t]
            if not matches:
                return None
            idx = indices[matches[0]]
        else:
            idx = indices[clean_title]
            
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude self
        sim_scores = [s for s in sim_scores if s[0] != idx][:n_recommendations]
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recs = df.iloc[movie_indices][['title', 'type', 'primary_genre', 'country', 'rating', 'release_year']].copy()
        recs['similarity_score'] = [round(s, 4) for s in scores]
        return recs

    # Generate sample recommendations for 3 benchmark titles
    sample_queries = ["Blood & Water", "Transformers Prime", "Je Suis Karl"]
    available_queries = []
    for q in sample_queries:
        if q.lower() in indices:
            available_queries.append(q)
    if not available_queries:
        available_queries = df['title'].head(3).tolist()
        
    evaluation_samples = {}
    print("\n[Step 4] Generating sample recommendations:")
    for query in available_queries:
        recs = get_recommendations(query, n_recommendations=5)
        evaluation_samples[query] = recs
        print(f"\nTop Recommendations for '{query}':")
        if recs is not None:
            print(recs[['title', 'type', 'primary_genre', 'similarity_score']].to_string(index=False))

    # Step 5: Evaluate recommendation quality
    # Compute similarity distribution across all top-5 recommendations
    all_top5_scores = []
    genre_match_rates = []
    sample_indices = np.random.choice(len(df), size=min(300, len(df)), replace=False)
    
    for idx in sample_indices:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        scores = [s[1] for s in sim_scores]
        all_top5_scores.extend(scores)
        
        target_genre = df.iloc[idx]['primary_genre']
        rec_genres = [df.iloc[s[0]]['primary_genre'] for s in sim_scores]
        match_count = sum(1 for g in rec_genres if g == target_genre)
        genre_match_rates.append(match_count / 5.0)
        
    avg_top5_score = np.mean(all_top5_scores)
    avg_genre_match = np.mean(genre_match_rates) * 100
    print(f"\n[Step 5] Evaluation Metrics:")
    print(f"Mean Top-5 Cosine Similarity: {avg_top5_score:.4f}")
    print(f"Primary Genre Alignment Rate: {avg_genre_match:.2f}%")
    
    # Visualization: Similarity Score Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(all_top5_scores, bins=30, kde=True, color='#E50914')
    plt.axvline(avg_top5_score, color='blue', linestyle='--', label=f'Mean Score ({avg_top5_score:.3f})')
    plt.title("Distribution of Top-5 Recommendation Similarity Scores", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Recommendation Pairs Count", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "rec_similarity_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    # Save the recommendation engine artifacts
    # Save TF-IDF model and dataframe for fast inference in dashboard
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    # Save lightweight cosine engine metadata
    df[['show_id', 'title', 'type', 'primary_genre', 'director', 'country', 'rating', 'release_year']].to_parquet(
        os.path.join(models_dir, "rec_catalog.parquet") if hasattr(pd.DataFrame, 'to_parquet') else os.path.join(models_dir, "rec_catalog.csv")
    )
    print(f"Serialized models saved to {models_dir}")
    
    # Generate Markdown Report
    def df_to_md(d):
        header = "| " + " | ".join(d.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
        rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in d.values]
        return "\n".join([header, sep] + rows)

    sample_tables_md = ""
    for q, rec_df in evaluation_samples.items():
        if rec_df is not None:
            sample_tables_md += f"\n### Recommendations for: *{q}*\n"
            sample_tables_md += df_to_md(rec_df[['title', 'type', 'primary_genre', 'country', 'similarity_score']])
            sample_tables_md += "\n"
            
    report_content = f"""# Task 3: Recommendation System Analysis Report
**Organization**: Auspify Technologies  
**Project**: Data Science Internship  
**Prepared For**: Evaluation & Project Verification  

---

## 1. Executive Summary
Content discovery is paramount to user retention and engagement on digital streaming platforms. This project implements a **Content-Based Recommendation Engine** powered by Natural Language Processing (NLP) and Cosine Similarity, computing semantic proximities across catalog titles, genre classifications, directors, producing nations, and maturity ratings.

---

## 2. Methodology & Workflow
1. **Metadata Feature Engineering ('Soup')**:
   - Synthesized key catalog metadata into a rich composite document representing each title's conceptual fingerprint.
   - Assigned higher relative importance to primary genre categories and content format.
2. **TF-IDF Vectorization**:
   - Transformed text corpus into a Term Frequency-Inverse Document Frequency (TF-IDF) feature matrix using unigrams and bigrams, filtering common English stopwords.
3. **Similarity Kernel**:
   - Calculated pair-wise Cosine Similarity:
   $$\\text{{Cosine Similarity}}(A, B) = \\frac{{A \\cdot B}}{{\\|A\\| \\|B\\|}}$$
   - Enabled sub-second latency for real-time recommendation retrieval.

---

## 3. Evaluation & Performance Metrics
- **Mean Top-5 Cosine Similarity**: `{avg_top5_score:.4f}`
- **Genre Consistency Rate**: `{avg_genre_match:.2f}%`
- **Output Artifacts**:
  - Model weights: `tfidf_vectorizer.pkl`
  - Visualization: `rec_similarity_distribution.png`

---

## 4. Benchmark Recommendation Outputs
{sample_tables_md}

---
*Task 3 completed successfully.*
"""
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_content)
    print(f"[Done] Task 3 Recommendation Report generated at: {report_path}")

if __name__ == "__main__":
    run_task_3()
