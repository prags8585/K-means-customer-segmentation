import argparse, pandas as pd, numpy as np
from datetime import datetime

def compute_rfm(df: pd.DataFrame):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    return rfm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to transactions CSV")
    ap.add_argument("--out", required=True, help="Output path for RFM features CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    needed = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic cleaning
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    rfm = compute_rfm(df)
    rfm.to_csv(args.out, index=False)
    print(f"Saved RFM features to {args.out}")

if __name__ == "__main__":
    main()
