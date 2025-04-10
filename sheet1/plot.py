import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import numpy as np

# Apply the default theme
sns.set_theme()

# Load an dataset
df = pd.read_csv("./solar_panel.csv")

df["Date"] = pd.to_datetime(
    df[["Year", "Month", "Day", "First Hour of Period"]].rename(
        columns={"First Hour of Period": "Hour"}
    )
)

df = df.drop("Day", axis="columns")
df = df.drop("Year", axis="columns")
df = df.drop("Month", axis="columns")
df = df.drop("Day of Year", axis="columns")
df = df.drop("First Hour of Period", axis="columns")


def task1(df):
    numeric_cols = df.select_dtypes(include="number").columns
    n = len(numeric_cols)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], ax=axes[i], bins=30)
        axes[i].set_title(f"Histogram of {col}")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("histograms.png")

    plt.close()


def task2(df):
    df["DateOnly"] = df["Date"].dt.date

    daily_sum = df.groupby("DateOnly").agg({"Power Generated": "sum"}).reset_index()
    idx = daily_sum["Power Generated"].idxmax()
    print(
        "The most power was generated on",
        daily_sum["DateOnly"][idx],
        "with a value of",
        daily_sum["Power Generated"][idx],
        ".",
    )
    ax = sns.relplot(data=daily_sum, x="DateOnly", y="Power Generated", kind="line")
    ax.tick_params(axis="x", labelrotation=45)
    plt.savefig("power.png")

    plt.close()


def task3(df):
    df["DateOnly"] = df["Date"].dt.date

    daily_avg = (
        df.groupby("DateOnly")
        .agg({"Sky Cover": "mean", "Power Generated": "sum"})
        .reset_index()
    )
    # clustering using kmeans

    kmeans = KMeans(n_clusters=2, random_state=42)
    daily_avg["Cluster"] = kmeans.fit_predict(daily_avg[["Sky Cover"]])

    ax = sns.relplot(data=daily_avg, x="DateOnly", y="Power Generated", hue="Cluster")
    ax.tick_params(axis="x", labelrotation=45)
    plt.savefig("sky_cover_manual.png")

    plt.close()
    # manual clustering

    conditions = [
        daily_avg["Sky Cover"] <= 2,
        daily_avg["Sky Cover"] > 2,
    ]

    choices = [0, 1]

    daily_avg["Cluster"] = np.select(conditions, choices)
    ax = sns.relplot(data=daily_avg, x="DateOnly", y="Power Generated", hue="Cluster")
    ax.tick_params(axis="x", labelrotation=45)
    plt.savefig("sky_cover_manual.png")

    plt.close()


task1(df)
task2(df)
task3(df)
