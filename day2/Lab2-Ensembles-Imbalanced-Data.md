# Lab 2: Ensembles and Imbalanced Data

> **Machine Learning with AI and Python** · Day 2
> Prerequisite: Lab 1 (Day 2) complete

## The scenario

You work at a subscription service and need to flag customers likely to cancel before they do. Only 12% of customers churn — the rest stay. If your model predicts "retained" for everyone, it scores 88% accuracy without learning a single pattern. This lab is about refusing to be fooled by that number.

## Why this lab matters

Lab 1 built one decision tree. A Random Forest builds hundreds of them, each on a random slice of data, and averages their votes — that is the ensemble idea. But ensembles inherit the same weakness as any model: when one class is rare, they learn to ignore it. You will see the trap, then work through three progressively different fixes — adjust the model's penalty, copy minority examples, and synthesize entirely new ones with SMOTE — and measure what each fix actually buys you.

## What you will do

- Generate and inspect an imbalanced customer dataset
- Fit a baseline Random Forest and watch high accuracy hide a broken recall
- Fix the imbalance three ways: class weights, random oversampling, and SMOTE
- Add Gradient Boosting and compare all approaches side by side
- Interpret the best model with feature importance
- Compare confusion matrices for baseline versus best model

## Before you start

You will use Python with scikit-learn, imbalanced-learn, and matplotlib. Create `ensembles_lab.py`.

Install imbalanced-learn if you have not already:

```
pip install imbalanced-learn
```

---

## Step 1: Generate and inspect the data

**Why:** `make_classification` lets you control the imbalance ratio exactly, which makes the problem transparent. Wrapping it in a DataFrame with meaningful column names makes the feature importance chart readable later. Always check the class split before modelling — it decides which metrics and fixes matter.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X_raw, y = make_classification(
    n_samples=3000, n_features=10, n_informative=7,
    weights=[0.88, 0.12], flip_y=0.02, random_state=42
)

feature_names = [
    "account_age_months", "monthly_spend",    "num_support_calls",
    "days_since_login",   "num_products",     "contract_months_left",
    "payment_failures",   "avg_session_mins", "num_complaints",  "discount_used"
]

X = pd.DataFrame(X_raw, columns=feature_names)

print("total customers:", len(y))
print("churned  (1)   :", y.sum(), f"({y.mean():.1%})")
print("retained (0)   :", (y == 0).sum(), f"({(1-y.mean()):.1%})")
```

**What to notice:** With an 88/12 split, a model that always predicts "retained" gets 88% accuracy without learning anything. That is the accuracy trap you will expose in the next step.

## Step 2: Split

**Why:** `stratify=y` keeps the same 88/12 ratio in both train and test. Without it, a random split can put too few churned customers in the test set to measure recall reliably.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape)
print("churn rate in test:", round(y_test.mean(), 3))
```

---

## Step 3: Baseline Random Forest — the accuracy trap

**Why:** A Random Forest trains hundreds of decision trees like the ones from Lab 1, each on a random subset of rows and features, then takes a majority vote. It is a strong general-purpose model. But on imbalanced data, the majority class dominates the vote and the minority class gets ignored.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, classification_report)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("accuracy:", round(accuracy_score(y_test, pred_rf), 3))
print("recall  :", round(recall_score(y_test, pred_rf), 3))
print("F1      :", round(f1_score(y_test, pred_rf), 3))
print()
print(classification_report(y_test, pred_rf, target_names=["retained", "churned"]))
```

**What to notice:** Accuracy looks strong, but recall on churned customers is low — the model is missing most of the customers you actually need to act on. A retention team working from these predictions would largely be flying blind.

> [!NOTE]
> **Checkpoint:** You can state the baseline recall on churned customers and explain why the accuracy number is misleading on its own.

## Step 4: Fix 1 — class weights

**Why:** `class_weight="balanced"` tells the model to penalize misclassifying the minority class more heavily during training. This is the cheapest fix: no extra data, no new library, just a parameter change. You saw this in Lab 4 on Day 1 with logistic regression — here it applies to a forest instead.

```python
rf_bal = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42
).fit(X_train, y_train)
pred_bal = rf_bal.predict(X_test)

print("baseline  recall:", round(recall_score(y_test, pred_rf),  3),
      "  F1:", round(f1_score(y_test, pred_rf),  3))
print("balanced  recall:", round(recall_score(y_test, pred_bal), 3),
      "  F1:", round(f1_score(y_test, pred_bal), 3))
```

> [!TIP]
> Class weights fix the model's objective, not the data. The forest still sees the same 88/12 ratio — it just cares more about getting the 12% right.

---

## Step 5: Fix 2 — random oversampling

**Why:** Instead of adjusting the model, you can fix the data. Random oversampling duplicates existing minority samples until both classes are equal. The model trains on a balanced dataset and naturally learns the minority class better. The downside: duplicates do not add new information.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)

print("before resampling:", np.bincount(y_train))
print("after resampling :", np.bincount(y_ros))

rf_ros = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ros, y_ros)
pred_ros = rf_ros.predict(X_test)

print("oversampled  recall:", round(recall_score(y_test, pred_ros), 3),
      "  F1:", round(f1_score(y_test, pred_ros), 3))
```

**What to notice:** The class counts are now equal in the training set. Recall should improve over the baseline, but since duplicates carry no new signal, results may plateau — which is the motivation for SMOTE.

## Step 6: Fix 3 — SMOTE

**Why:** SMOTE (Synthetic Minority Over-sampling TEchnique) goes further than duplication. For each minority sample, it finds its nearest neighbors in feature space and generates a new synthetic example somewhere between them. The model trains on genuinely new examples, not just copies.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

print("after SMOTE:", np.bincount(y_sm))

rf_sm = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_sm, y_sm)
pred_sm = rf_sm.predict(X_test)

print("SMOTE  recall:", round(recall_score(y_test, pred_sm), 3),
      "  F1:", round(f1_score(y_test, pred_sm), 3))
```

> [!TIP]
> SMOTE synthesizes new samples in the *training set only*. The test set stays untouched — you always evaluate on real, unmodified data.

---

## Step 7: Add Gradient Boosting and compare everything

**Why:** Gradient Boosting builds trees sequentially — each one focuses on the mistakes of the last. It often outperforms a Random Forest on structured tabular data because it applies training effort where the model is currently weakest. Combine it with SMOTE and put all approaches in one table.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_sm = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_sm, y_sm)
pred_gb = gb_sm.predict(X_test)

results = [
    ("RF baseline",       pred_rf),
    ("RF + class_weight", pred_bal),
    ("RF + oversampling", pred_ros),
    ("RF + SMOTE",        pred_sm),
    ("GBM + SMOTE",       pred_gb),
]

print(f"{'model':<22} {'recall':>8} {'F1':>8} {'accuracy':>10}")
print("-" * 52)
for name, pred in results:
    print(f"{name:<22} {recall_score(y_test, pred):>8.3f} "
          f"{f1_score(y_test, pred):>8.3f} "
          f"{accuracy_score(y_test, pred):>10.3f}")
```

**What to notice:** As recall rises, accuracy often dips slightly — that is the precision cost of catching more churned customers. The goal is not to maximize any single column but to pick the row that fits your stated business constraint.

> [!NOTE]
> **Checkpoint:** You can read this table and say which model you would deploy if missing a churned customer is more costly than a false alarm, and justify it by pointing to a specific number.

---

## Step 8: Feature importance

**Why:** The best model should also be the most trusted model, and trust requires explanation. Feature importance shows which customer signals the forest used to flag churn risk — and which ones were essentially noise.

```python
import matplotlib.pyplot as plt

importances = rf_sm.feature_importances_
order = np.argsort(importances)[::-1]

print("feature importance ranking:")
for i in order:
    bar = "#" * int(importances[i] * 60)
    print(f"  {feature_names[i]:<24} {importances[i]:.3f}  {bar}")

plt.figure(figsize=(8, 4))
plt.bar(range(len(importances)), importances[order])
plt.xticks(range(len(importances)),
           [feature_names[i] for i in order], rotation=30, ha="right")
plt.ylabel("importance")
plt.title("Feature importance — RF + SMOTE")
plt.tight_layout()
plt.show()
```

**What to notice:** Features at the bottom are doing almost nothing. Dropping them would give you a simpler, faster model that is easier to explain to a product or business team.

## Step 9: Confusion matrices — baseline versus best

**Why:** The summary table shows numbers; the confusion matrix shows where the errors land. Comparing baseline to best side by side makes the improvement concrete and shows exactly what you are trading away in false positives to gain recall.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, pred, title in [
    (axes[0], pred_rf, "RF baseline"),
    (axes[1], pred_sm, "RF + SMOTE"),
]:
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred),
        display_labels=["retained", "churned"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

**What to notice:** The bottom-left cell (predicted retained, actually churned) is the costly error — customers you failed to catch. Watch how it shrinks from baseline to SMOTE. The bottom-right cell (false alarms) will grow as the trade-off.

> [!NOTE]
> **Checkpoint:** You can point to the cell that represents missed churners and say by how many it shrank — and what it cost in false alarms.

---

## On your own

You are the analyst recommending a model to the retention team:

1. The team can only call 50 customers per week. Does that constraint push you toward higher recall or higher precision? Pick the model row from the comparison table that best fits that budget and explain why.
2. Change `weights=[0.88, 0.12]` in Step 1 to `weights=[0.97, 0.03]` for a more extreme imbalance. Re-run the lab. Which fix holds up best, and which breaks down first?
3. Write two sentences: which combination of model and imbalance fix you would ship, and one thing you would want to verify about the real data before trusting these results.

## Responsible AI

> [!IMPORTANT]
> - Synthetic data hides real-world bias. A churn model trained on historical data can encode past discrimination if certain customer groups were treated differently — check error rates by group before deployment.
> - SMOTE generates plausible examples, not real ones. Always evaluate on held-out real data, never on any resampled portion of the training set.
> - Recall and precision pull in opposite directions. Be explicit about which error costs more before choosing a threshold or resampling strategy, and put that reasoning in writing where stakeholders can see it.

## What you learned

- Identify the accuracy trap on imbalanced data and use recall and F1 instead
- Fix imbalance three ways: class weights (model-level), random oversampling (data duplication), and SMOTE (synthetic minority samples)
- Understand how Random Forests and Gradient Boosting differ as ensemble strategies
- Compare multiple approaches in a single table and choose based on a stated goal
- Explain a model's decisions with feature importance and read a confusion matrix to find the costly errors

## Stretch goals

- Try `imblearn.over_sampling.ADASYN` as a drop-in replacement for SMOTE and compare results — does it consistently outperform, or does it depend on the imbalance ratio?
- Use `imblearn.pipeline.Pipeline` (not sklearn's) to chain SMOTE and a classifier so resampling only happens *inside* each cross-validation fold. Compare those scores to the approach above and explain why they may differ.
- Tune `GradientBoostingClassifier` with `GridSearchCV(scoring="f1")` across `learning_rate` (0.05, 0.1, 0.2) and `max_depth` (2, 3, 4). Does tuning close the gap with RF + SMOTE, or does RF hold its lead?
