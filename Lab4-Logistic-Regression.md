# Lab 4: Building and Judging a Classifier with Logistic Regression

> **Machine Learning with AI and Python** · Day 1
> Prerequisite: Lab 3 complete

## The scenario

You are building a screening tool: it flags cases for a human to review, it does not make the final call. The data is a medical screening set (each row is a tumor, and you must catch the malignant ones), but the pattern is the same for fraud, churn, and loan review: one outcome is rare and costly to miss. The lesson of this lab is not the code, which mirrors Lab 3, it is judging a classifier responsibly, because accuracy alone can hide dangerous mistakes.

## Why this lab matters

The slides covered probability and a threshold, the accuracy trap, and precision versus recall. A real classification project adds the habits that keep a model honest: cross-validation, a threshold chosen for a stated goal, handling class imbalance, comparing against another model, understanding what drives predictions, and saving the result. You will do all of it.

## What you will do

- Load and frame the problem, then check the class balance
- Fit a baseline pipeline and read probabilities, not just labels
- Judge it with a confusion matrix, precision, recall, F1, ROC/AUC, and a PR curve
- Validate with cross-validation, then choose a threshold for a goal
- Handle imbalance, compare against another model, and inspect what drives predictions
- Save the model

## Before you start

You will use Python with scikit-learn and matplotlib. Create `classification_lab.py`. This lab uses the breast cancer dataset that ships with scikit-learn, so there is no download.

---

## Step 1: Load, frame, and check the balance

**Why:** The dataset encodes benign as `1` and malignant as `0`. Because the case we must catch is malignant, we flip the labels so `1` means malignant. That way *recall* measures how many real cancers we caught. Always check the class balance first: it decides which metrics matter.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X = data.data
y = (data.target == 0).astype(int)     # 1 = malignant (the case to catch)

print("rows, features:", X.shape)
print("malignant rate:", round(y.mean(), 3))   # the positive class share
```

### Step 2: Split, keeping the class balance

**Why:** `stratify=y` keeps the same proportion of malignant cases in train and test. Without it, a random split can leave too few positives in the test set to measure recall reliably.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## Step 3: Fit a baseline and read probabilities

**Why:** Scaling helps logistic regression converge; a `Pipeline` learns the scaling from training data only. `predict` gives the label; `predict_proba` gives the probability behind it, which is what lets you move the threshold later.

```python
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000)).fit(X_train, y_train)

pred  = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]   # probability of malignant
print(proba[:10].round(3))
```

## Step 4: Judge it properly

**Why:** When positives are rare, accuracy hides misses. The confusion matrix, precision, recall, and F1 expose the errors accuracy averages away, and AUC summarizes performance across all thresholds.

```python
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

print(confusion_matrix(y_test, pred))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("F1       :", round(f1_score(y_test, pred), 3))
print("ROC AUC  :", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred, target_names=["benign", "malignant"]))
```

**How to read these:** Recall is "of the real malignant cases, how many did we catch?" (a missed cancer is the costly error). Precision is "of the ones we flagged, how many were real?" (false alarms). F1 balances the two. AUC near 1.0 means the ranking of cases by risk is strong.

> [!NOTE]
> **Checkpoint:** You can read the confusion matrix and say exactly how many malignant cases the model missed.

## Step 5: Validate with cross-validation

**Why:** One split can mislead. Cross-validation on the metric you care about (recall) shows the honest range you can expect.

```python
recall_scores = cross_val_score(model, X, y, cv=5, scoring="recall")
print("fold recall:", recall_scores.round(3))
print("mean recall:", round(recall_scores.mean(), 3), "+/-", round(recall_scores.std(), 3))
```

---

## Step 6: See the trade-off, then pick a threshold for a goal

**Why:** The default cutoff is `0.5`, but the right cutoff depends on the cost of a miss. Sweep the threshold, look at the curve, then choose on purpose. For screening, catching cases (recall) matters more than a few false alarms.

```python
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (proba >= t).astype(int)
    print(f"thr {t:.1f}  recall {recall_score(y_test, p):.3f}  "
          f"precision {precision_score(y_test, p):.3f}")
```

Now pick the highest threshold that still catches at least 99% of malignant cases, and see the precision you pay:

```python
for t in np.arange(0.50, 0.0, -0.01):
    p = (proba >= t).astype(int)
    if recall_score(y_test, p) >= 0.99:
        print(f"chosen threshold {t:.2f}: recall {recall_score(y_test, p):.3f}, "
              f"precision {precision_score(y_test, p):.3f}")
        break
```

> [!NOTE]
> **Checkpoint:** You can state a threshold, the recall it achieves, and the precision cost, and argue whether that trade is acceptable for a screening tool.

## Step 7 (optional): Plot the curves

**Why:** A picture makes the trade-off concrete. The precision-recall curve is more informative than ROC when one class is rare.

```python
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

PrecisionRecallDisplay.from_predictions(y_test, proba)
plt.title("Precision-recall")
plt.show()

RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC curve")
plt.show()
```

---

## Step 8: Handle class imbalance

**Why:** When positives are rare, a model can score well by ignoring them. `class_weight="balanced"` tells the model to care more about the rare class. Compare it to the baseline on recall.

```python
balanced = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced")
).fit(X_train, y_train)

print("baseline recall:", round(recall_score(y_test, pred), 3))
print("balanced recall:", round(recall_score(y_test, balanced.predict(X_test)), 3))
```

> [!TIP]
> `class_weight="balanced"` and moving the threshold both trade precision for recall. Reach for `class_weight` during training; use the threshold to tune the final decision for your goal.

## Step 9: Compare against another model

**Why:** Logistic regression is a strong, explainable baseline. Comparing to a different model type tells you whether more complexity is worth it, judged on the metric you care about.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=42).fit(X_train, y_train)
fp = forest.predict(X_test)
print("logistic  recall:", round(recall_score(y_test, pred), 3),
      "F1:", round(f1_score(y_test, pred), 3))
print("forest    recall:", round(recall_score(y_test, fp), 3),
      "F1:", round(f1_score(y_test, fp), 3))
```

## Step 10: Inspect what drives predictions

**Why:** A model that affects people should be explainable. For logistic regression the coefficients (on scaled features) show which measurements push a case toward "malignant."

```python
logreg = model.named_steps["logisticregression"]
weights = logreg.coef_[0]
top = np.argsort(np.abs(weights))[::-1][:5]
for i in top:
    print(f"{data.feature_names[i]:<24} weight {weights[i]:+.3f}")
```

**What to notice:** A positive weight pushes toward malignant, negative toward benign. This is the kind of explanation a reviewer or auditor will ask for.

> [!NOTE]
> **Checkpoint:** You can name the top few features driving the model and the direction each pushes.

## Step 11: Save the model

**Why:** Save the fitted pipeline (scaler included) so predictions elsewhere use the exact same preprocessing.

```python
import joblib
joblib.dump(model, "screening_model.joblib")
loaded = joblib.load("screening_model.joblib")
print("loaded recall:", round(recall_score(y_test, loaded.predict(X_test)), 3))
```

## On your own

Act as the reviewer for this screening tool:

1. Choose a threshold for a stated goal (for example, "miss no more than 1 in 50 malignant cases") and report the precision cost.
2. Compare the balanced logistic model and the random forest on recall using `cross_val_score(scoring="recall")`, not a single split.
3. Write two sentences: which model you would deploy behind a human reviewer, and one fairness or safety concern you would raise before it goes live.

## Responsible AI

> [!IMPORTANT]
> - A model that affects people needs more than a good accuracy number. Check that errors do not fall unfairly on one group.
> - Keep a human in the loop for consequential decisions. This tool flags cases for a clinician; it does not diagnose.
> - Report recall and precision (and the threshold you chose) alongside accuracy, so the real trade-offs are visible.

## What you learned

- Frame a rare-positive problem, check class balance, and stratify the split
- Judge a classifier with a confusion matrix, precision, recall, F1, ROC/AUC, and a PR curve
- Validate with cross-validation, then choose a threshold for a real goal
- Handle imbalance with `class_weight`, compare models, and explain predictions with coefficients

## Stretch goals

- Add a `LogisticRegression` without scaling and confirm whether it still converges within `max_iter`.
- Use `cross_val_score` with `scoring="f1"` and see whether it picks a different "best" model than recall did.
- Try `GradientBoostingClassifier`; does it beat logistic regression on recall enough to justify the loss of interpretability?
