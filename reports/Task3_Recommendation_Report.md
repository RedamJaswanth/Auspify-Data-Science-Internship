# Task 3: Recommendation System Analysis Report
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
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
   - Enabled sub-second latency for real-time recommendation retrieval.

---

## 3. Evaluation & Performance Metrics
- **Mean Top-5 Cosine Similarity**: `0.6985`
- **Genre Consistency Rate**: `94.07%`
- **Output Artifacts**:
  - Model weights: `tfidf_vectorizer.pkl`
  - Visualization: `rec_similarity_distribution.png`

---

## 4. Benchmark Recommendation Outputs

### Recommendations for: *Blood & Water*
| title | type | primary_genre | country | similarity_score |
| --- | --- | --- | --- | --- |
| 46 | TV Show | International TV Shows | Turkey | 0.7743 |
| On Children | TV Show | International TV Shows | Taiwan | 0.755 |
| We Are the Wave | TV Show | International TV Shows | Germany | 0.7424 |
| Katla | TV Show | International TV Shows | Iceland | 0.7396 |
| Between | TV Show | International TV Shows | Canada | 0.739 |

### Recommendations for: *Transformers Prime*
| title | type | primary_genre | country | similarity_score |
| --- | --- | --- | --- | --- |
| Dinotrux | TV Show | Kids' TV | United States | 0.8344 |
| Transformers: Robots in Disguise | TV Show | Kids' TV | United States | 0.8289 |
| Transformers: Cyberverse | TV Show | Kids' TV | Pakistan | 0.7962 |
| Sharkdog | TV Show | Kids' TV | United States | 0.7838 |
| Gabby's Dollhouse | TV Show | Kids' TV | United States | 0.7838 |

### Recommendations for: *Je Suis Karl*
| title | type | primary_genre | country | similarity_score |
| --- | --- | --- | --- | --- |
| Lommbock | Movie | Comedies | Germany | 0.6221 |
| Baazaar | Movie | Dramas | India | 0.4785 |
| Ajji | Movie | Dramas | India | 0.4785 |
| Samantaral | Movie | Dramas | India | 0.4785 |
| Deranged | Movie | Dramas | Unknown | 0.4746 |


---
*Task 3 completed successfully.*
