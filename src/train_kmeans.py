# src/train_kmeans.py
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")


def find_best_k(X, max_k=10):
    """Find best number of clusters based on silhouette score."""
    n_samples = len(X)
    if n_samples < 5:
        # Not enough samples for silhouette-based selection
        return 1

    best_k = 2
    best_score = -1
    for k in range(2, min(max_k, n_samples) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:  # Need at least 2 clusters
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def main():
    parser = argparse.ArgumentParser(description="Train KMeans on RFM data")
    parser.add_argument("--input", required=True, help="Input CSV with RFM features")
    parser.add_argument("--k", type=int, help="Optional: number of clusters")
    parser.add_argument("--out", required=True, help="Output CSV with cluster labels")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    n_samples = len(df)

    if n_samples < 2:
        print("Not enough samples for clustering. Saving data as-is.")
        df["Cluster"] = 0
        df.to_csv(args.out, index=False)
        return

    # Prepare features
    X = df[["Recency", "Frequency", "Monetary"]].values
    Xs = StandardScaler().fit_transform(X)

    # Choose k
    if args.k:
        best_k = min(args.k, n_samples)
    else:
        print("Finding best k based on silhouette score...")
        best_k = find_best_k(Xs)

    if best_k < 2:
        print("Dataset too small for clustering. Assigning all to one cluster.")
        df["Cluster"] = 0
        df.to_csv(args.out, index=False)
        print(f"Clustered data saved to {args.out}")
        return

    print(f"Using k={best_k} clusters")

    # Train KMeans
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(Xs)

    # Save
    df.to_csv(args.out, index=False)
    print(f"Clustered data saved to {args.out}")


if __name__ == "__main__":
    main()
