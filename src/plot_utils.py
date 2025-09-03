import matplotlib.pyplot as plt
import pandas as pd

def hist_feature(df: pd.DataFrame, col: str, outpath: str):
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def elbow_plot(distortions, ks, outpath):
    plt.figure()
    plt.plot(ks, distortions, marker="o")
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
