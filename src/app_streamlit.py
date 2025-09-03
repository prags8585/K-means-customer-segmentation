import streamlit as st
import pandas as pd
from io import StringIO
from data_prep import compute_rfm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="RFM Customer Segmentation", layout="wide")

st.title("🧩 RFM Customer Segmentation")
uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    try:
        rfm = compute_rfm(df)
        st.subheader("RFM Features")
        st.dataframe(rfm.head())

        k = st.slider("Number of clusters (k)", 2, 10, 4)
        X = rfm[["Recency","Frequency","Monetary"]].values
        Xs = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(Xs)
        rfm["cluster"] = labels

        st.subheader("Clustered Customers")
        st.dataframe(rfm.sort_values("cluster").head(50))

        st.download_button("Download labeled customers CSV", rfm.to_csv(index=False), "clustered_customers.csv", "text/csv")

        st.bar_chart(rfm.groupby("cluster")["Monetary"].mean(), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV with columns: InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice")
