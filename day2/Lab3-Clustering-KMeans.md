# Lab 3: Finding Structure Without Labels — K-Means Clustering

> **Machine Learning with AI and Python** · Day 2
> Prerequisite: Lab 2 (Day 2) complete

## The scenario

You have measurements for 150 iris flowers but no species labels. Can an algorithm discover the natural groupings on its own — purely from the numbers? This is unsupervised learning: no right answers to train on, no recall to optimize. The goal is to find structure that is already in the data. The same idea powers customer segmentation, document grouping, anomaly detection, and genomics. The lesson is not just the algorithm — it is learning to evaluate a model when you have nothing to check it against.

## Why this lab matters

Every lab so far has had labels: a price to predict, a diagnosis to classify, a customer outcome to flag. Unsupervised learning removes that scaffold entirely. K-means is the simplest clustering algorithm and the right starting point. You will also meet two ideas that carry across all of unsupervised learning: the elbow method for choosing the number of clusters, and the silhouette score for evaluating quality without ground truth. At the end, you will reveal the true labels and see how closely the algorithm recovered them — that reveal is the point.

## What you will do

- Load iris and set the labels aside, unseen
- Scale the features (required for distance-based algorithms)
- Use the elbow method to choose the number of clusters
- Fit K-means and visualize the result
- Evaluate with silhouette score — no labels needed
- Reveal the true species labels and compare with a crosstab
- Try different values of k and confirm your choice
- Inspect per-cluster feature means to understand what each group represents

## Before you start

You will use Python with scikit-learn, pandas, and matplotlib. Create `clustering_lab.py`. This lab uses the iris dataset that ships with scikit-learn, so there is no download.

---

## Step 1: Load iris and set the labels aside

**Why:** The experiment is to see what the algorithm can find without any label information. Load the data, print the shape and feature names so you know what you are working with, and store the true labels in a separate variable that you will not touch until the reveal in Step 7.

```python
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data.data
true_labels = data.target          # set aside — do not use until Step 7

print("rows, features:", X.shape)
print("features:", data.feature_names)
print("(labels are hidden — the algorithm will not see them)")
```

## Step 2: Scale the features

**Why:** K-means assigns points to clusters by measuring distance. If one feature is in centimetres and another in millimetres, the larger-scale feature will dominate every distance calculation. Scaling puts all features on equal footing. This is the opposite of decision trees (Lab 1), which split on one feature at a time and are immune to scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("mean before scaling:", X.mean(axis=0).round(2))
print("mean after  scaling:", X_scaled.mean(axis=0).round(2))   # should be ~0
print("std  after  scaling:", X_scaled.std(axis=0).round(2))    # should be ~1
```

---

## Step 3: Elbow method — how many clusters?

**Why:** K-means needs you to choose k before fitting. The elbow method fits the model for a range of k values and plots the inertia (total squared distance from each point to its cluster centre). Inertia always drops as k grows, but the rate of improvement slows at the natural number of clusters — that slowdown looks like an elbow in the curve.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(k_range, inertias, marker="o")
plt.xlabel("number of clusters (k)")
plt.ylabel("inertia")
plt.title("Elbow method — look for the bend")
plt.xticks(k_range)
plt.tight_layout()
plt.show()
```

**What to notice:** The curve drops steeply from k=1 to k=3 and then flattens. The elbow at k=3 suggests three natural groups — which happens to match the three iris species, but the algorithm does not know that yet.

> [!TIP]
> The elbow is a heuristic, not a theorem. Sometimes the bend is obvious; sometimes it is gradual and you need the silhouette score from Step 6 to confirm your choice.

## Step 4: Fit K-means with k=3

**Why:** With k chosen, fit the model on the scaled data. `n_init=10` runs the algorithm ten times with different random starting points and keeps the best result — K-means is sensitive to initialisation, so this is standard practice.

```python
km = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = km.fit_predict(X_scaled)

print("cluster assignments (first 10):", clusters[:10])
print("cluster sizes:", np.bincount(clusters))
```

---

## Step 5: Visualize the clusters

**Why:** A scatter plot makes the structure concrete. Petal length and petal width were the most important features in Lab 1 — they separate the species most cleanly, so they are the right axes to plot here too.

```python
petal_length_idx = 2   # "petal length (cm)"
petal_width_idx  = 3   # "petal width (cm)"

colors = ["steelblue", "tomato", "seagreen"]

plt.figure(figsize=(7, 5))
for c in range(3):
    mask = clusters == c
    plt.scatter(X[mask, petal_length_idx], X[mask, petal_width_idx],
                color=colors[c], label=f"cluster {c}", alpha=0.7, edgecolors="k", linewidths=0.4)

centres = scaler.inverse_transform(km.cluster_centers_)
plt.scatter(centres[:, petal_length_idx], centres[:, petal_width_idx],
            marker="X", s=180, color="black", zorder=5, label="centres")

plt.xlabel(data.feature_names[petal_length_idx])
plt.ylabel(data.feature_names[petal_width_idx])
plt.title("K-means clusters (k=3) — petal features")
plt.legend()
plt.tight_layout()
plt.show()
```

**What to notice:** One cluster should be clearly separated. Two will overlap slightly — you will see exactly where in Step 7.

> [!NOTE]
> **Checkpoint:** You can describe the visual separation between clusters and point to the region where two groups blur together.

## Step 6: Evaluate without ground truth — silhouette score

**Why:** When you have no labels, you cannot measure recall or accuracy. The silhouette score measures two things for each point: how close it is to the other points in its own cluster (cohesion), and how far it is from the nearest other cluster (separation). A score near 1.0 means tight, well-separated clusters. Near 0 means clusters overlap. Negative means a point may be in the wrong cluster.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, clusters)
print(f"silhouette score (k=3): {score:.3f}")
print()
print("reading: 1.0 = perfect separation, 0.0 = overlapping, <0 = misassigned")
```

**What to notice:** A score around 0.55 for k=3 on iris is typical and indicates well-formed clusters. You will use this number in Step 8 to compare against other values of k.

---

## Step 7: The reveal — compare clusters to true species

**Why:** You can now check how well the algorithm recovered structure it was never told about. A crosstab counts how many flowers of each true species landed in each cluster. Perfect recovery would put all 50 of each species in a single cluster — see how close K-means gets.

```python
import pandas as pd

reveal = pd.DataFrame({
    "cluster": clusters,
    "species": [data.target_names[t] for t in true_labels]
})

print(pd.crosstab(reveal["cluster"], reveal["species"]))
```

**What to notice:** Setosa lands almost entirely in one cluster — its petal measurements are so distinct that no algorithm struggles with it. Versicolor and virginica share a cluster boundary, which mirrors the visual overlap from Step 5. The algorithm recovered the species grouping using only measurements, with no label information at all.

> [!NOTE]
> **Checkpoint:** You can read the crosstab and say which species is perfectly separated, which two overlap, and roughly how many flowers are misassigned.

## Step 8: Try k=2 and k=4, confirm the choice

**Why:** The elbow suggested k=3, but good practice is to check neighbours. Comparing silhouette scores for k=2, 3, and 4 either confirms the choice or reveals a better option.

```python
for k in [2, 3, 4]:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, labels_k)
    print(f"k={k}  silhouette: {s:.3f}")
```

**What to notice:** k=3 should score highest, confirming the elbow. If k=2 were higher, it would suggest the algorithm finds only two meaningfully distinct groups in the data — a useful finding in its own right.

---

## Step 9: What drove the clusters — per-cluster feature means

**Why:** Cluster labels (0, 1, 2) are arbitrary numbers. To make them meaningful you need to look at the feature means for each group. This is how you turn "cluster 2" into "the small-petal group" — something you could explain to a botanist or a business stakeholder.

```python
df = pd.DataFrame(X, columns=data.feature_names)
df["cluster"] = clusters

print(df.groupby("cluster")[list(data.feature_names)].mean().round(2))
```

**What to notice:** The clusters should separate cleanly on petal measurements (long/wide → one species, short/narrow → another), confirming the same pattern seen in feature importance during Lab 1. Sepal measurements will differ less across clusters.

> [!NOTE]
> **Checkpoint:** You can give a plain-language description of each cluster based on its feature means — the kind of summary a non-technical stakeholder could act on.

---

## On your own

You are presenting these clusters to a botanical survey team:

1. The team asks why you scaled the data before clustering but not before the decision tree in Lab 1. Write two sentences explaining the difference.
2. Try `KMeans(n_clusters=3, init="random", n_init=1, random_state=0)` and compare the cluster assignments to the standard result. What happens, and why does `n_init=10` exist?
3. Write a plain-English description of each cluster — no numbers, just the kind of language a field botanist would use — based on the feature means from Step 9.

## Responsible AI

> [!IMPORTANT]
> - Clustering has no ground truth, so it is easy to over-trust the result. Always ask whether the clusters reflect real structure or just the algorithm's assumptions about shape and distance.
> - K-means assumes roughly spherical, equal-sized clusters. If the real groups are elongated, nested, or very different in size, K-means will mislead you. Visualize before concluding.
> - In practice, cluster labels get attached to people (customer segments, patient groups). Check that the groupings do not encode or amplify demographic bias before acting on them.

## What you learned

- Frame an unsupervised problem: work without labels and evaluate without ground truth
- Scale features before any distance-based algorithm, and understand why trees do not need this
- Use the elbow method to choose k and the silhouette score to confirm it
- Visualize clusters and read per-cluster feature means to give groups meaning
- Verify discovered structure against known labels when they are eventually available

## Stretch goals

- Try `DBSCAN` from `sklearn.cluster` on the scaled iris data. It does not need k — it finds clusters by density. Compare its cluster assignments to K-means on the crosstab.
- Plot the silhouette score for every point (not just the mean) using `sklearn.metrics.silhouette_samples` and color by cluster. Which cluster has the most borderline members?
- Apply K-means to the wine dataset (`load_wine`) with labels hidden and see whether three clusters recover the three wine cultivars as cleanly as they did for iris.
