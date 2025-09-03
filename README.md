# Customer Segmentation with RFM + K-Means

customer segmentation using **RFM (Recency, Frequency, Monetary)** features and **K-Means** clustering.

## What this repo contains
- `src/data_prep.py` — Cleans transactions and computes RFM features.
- `src/train_kmeans.py` — Trains a KMeans model (with elbow & silhouette diagnostics) and saves results.
- `src/plot_utils.py` — Helper plotting functions.
- `src/app_streamlit.py` — A minimal Streamlit app to upload your CSV and visualize segments.
- `data/raw/sample_transactions.csv` — A small synthetic dataset to try locally.
- `requirements.txt` — Python dependencies.
- `Makefile` — Common tasks.

## Quickstart
```bash
# 1) Create env & install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run data prep (uses sample data by default)
python src/data_prep.py --input data/raw/sample_transactions.csv --out data/rfm_features.csv

# 3) Train KMeans and produce plots + labeled customers
python src/train_kmeans.py --input data/rfm_features.csv --k 4 --out data/clustered_customers.csv

# 4) (Optional) Streamlit UI
streamlit run src/app_streamlit.py
```

## Input schema
The project expects a CSV with columns similar to the UCI Online Retail dataset:
- `InvoiceNo` (string/int)
- `InvoiceDate` (datetime string)
- `CustomerID` (string/int)
- `Quantity` (int)
- `UnitPrice` (float)
- `Country` (string)

## Outputs
- `data/rfm_features.csv` — R, F, M features per customer.
- `data/clustered_customers.csv` — Cluster label & features for each customer.
- `reports/figures/` — Elbow and silhouette diagnostics, feature distributions.

## Notes
- Replace the sample CSV with your real data for better results.
- Choose `--k` via elbow/silhouette guidance.
