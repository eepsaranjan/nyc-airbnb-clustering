# nyc-airbnb-clustering
# Airbnb NYC 2019 — Listing Segmentation using Clustering

**Type:** Unsupervised Learning (Clustering)
**Contribution:** Individual

## Project Summary

The New York City Airbnb Open Data (2019) contains close to 49,000 listings across the five boroughs, with information on host, location, room type, price, minimum nights, review activity, and availability. Since there are no pre-existing labels describing "what kind of listing" each one is, this project uses unsupervised learning to group the listings into a small number of meaningful segments — so hosts, Airbnb's growth team, and prospective guests can quickly understand a listing's profile without reading every column.

## Problem Statement

Use unsupervised learning — **K-Means**, **Hierarchical (Agglomerative) Clustering**, and **DBSCAN** — on each listing's location, price, availability, and review-activity features to discover natural groupings. The resulting clusters should be interpretable enough to support pricing guidance, host-support strategy, and supply-growth decisions.

## Dataset

- **File:** `AB_NYC_2019.csv` (place in the same folder as the notebook, or update the `path` variable; a Google Drive mount option is included for Colab users)
- **Size:** 48,895 rows × 16 columns
- **Key fields:** `neighbourhood_group` (borough), `neighbourhood`, `latitude`/`longitude`, `room_type`, `price`, `minimum_nights`, `number_of_reviews`, `last_review`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`

## Workflow

1. **Know Your Data** — shape, info, duplicates, missing values
2. **Understanding Your Variables** — column descriptions, unique value counts
3. **Data Wrangling** — impute `reviews_per_month` with 0 (no reviews yet), fill missing `name`/`host_name`, drop `last_review`, remove invalid `price == 0` rows
4. **EDA & Visualization** — 15 charts covering price distribution, borough supply/pricing, room-type mix, geographic scatter plots, minimum nights, review activity, availability, host concentration, a listing-title word cloud, a correlation heatmap, and a pair plot
5. **Feature Engineering & Preprocessing**
   - Outlier handling: 99th-percentile winsorizing of `price` and `minimum_nights`
   - Categorical encoding: one-hot encoding of `neighbourhood_group` and `room_type`
   - Text preprocessing on listing titles: contraction expansion, lowercasing, punctuation/URL/digit removal, stopword removal, abbreviation normalization, tokenization, lemmatization, POS tagging, and TF-IDF vectorization (used for exploratory insight only, not fed into clustering)
   - Engineered features: `price_per_review`, `reviews_per_listing`, `availability_ratio`
   - Feature scaling with `StandardScaler`
   - Dimensionality reduction with PCA (2 components) for visualization only
   - 80/20 train/test split used as a stability check, not for supervised evaluation
6. **Modeling** — three clustering algorithms compared:
   - **K-Means** — tuned over k = 2–10 via the elbow method plus silhouette/Davies-Bouldin scoring; **k = 5** selected
   - **Agglomerative (Hierarchical) Clustering** — fit on a sample; linkage methods (`ward`, `complete`, `average`) compared, with `ward` performing best
   - **DBSCAN** — grid-searched over `eps` and `min_samples`; flags sparse/isolated listings as noise instead of forcing them into a cluster
7. **Evaluation** — Silhouette Score and Davies-Bouldin Index used throughout, since clustering has no ground-truth labels to score against
8. **Final Model** — **K-Means (k = 5)** was selected: best silhouette/Davies-Bouldin balance, scales to the full dataset, produces five clearly interpretable segments, and its stability check confirmed the structure holds on unseen data. Clusters were profiled by comparing each segment's feature means to the overall average.

## Key Findings

- Price is heavily right-skewed, concentrated in the $50–$200/night range
- Supply is concentrated in Manhattan and Brooklyn; price broadly tracks distance from Manhattan
- A small number of hosts manage a disproportionate number of listings, pointing to a professional/commercial host segment
- The five K-Means segments broadly correspond to profiles such as popular entire-home listings in Brooklyn, higher-priced low-turnover professional listings in Manhattan, and other distinct price/location/activity combinations

## Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **NLP:** `nltk` (stopwords, tokenization, lemmatization, POS tagging), `scikit-learn` (`TfidfVectorizer`, `TruncatedSVD`)
- **Modeling & evaluation:** `scikit-learn` (`KMeans`, `AgglomerativeClustering`, `DBSCAN`, `StandardScaler`, `PCA`, `silhouette_score`, `davies_bouldin_score`, `train_test_split`)

## How to Run

1. Install dependencies: `pandas numpy matplotlib seaborn wordcloud nltk scikit-learn`
2. Place `AB_NYC_2019.csv` in the same directory as the notebook (or mount Google Drive and update the `path` variable if running in Colab)
3. Run all cells top to bottom — required NLTK corpora are downloaded automatically at the start of the notebook

## Notes

- No target variable exists for this problem; all hyperparameter tuning and model comparison rely on internal clustering metrics (Silhouette Score, Davies-Bouldin Index) rather than supervised cross-validation
