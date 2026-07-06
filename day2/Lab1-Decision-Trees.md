# Lab 1: Understanding Decisions with Decision Trees

> **Machine Learning with AI and Python** · Day 2
> Prerequisite: Day 1 complete

## The scenario

A botanist wants a tool that classifies iris flowers by species from four measurements: sepal length, sepal width, petal length, and petal width. The pattern is the same for any problem where you need a rule humans can read and verify — credit approval, medical triage, equipment fault detection. Decision trees are the rare model you can print on a page and hand to a domain expert.

## Why this lab matters

The slides covered how trees split data and what Gini impurity measures. A real tree project adds the habits that keep a model honest: catching overfit before it ships, choosing a depth that generalizes, reading the tree to verify the rules make sense, and knowing which features are doing the work. You will do all of it.

## What you will do

- Load and explore the iris dataset
- Fit a fully grown tree and witness it overfit
- Limit the depth and confirm the fix
- Visualize the tree and read its decisions
- Understand what Gini impurity means in each node
- Plot feature importance and drop weak features
- Judge the result with a confusion matrix

## Before you start

You will use Python with scikit-learn and matplotlib. Create `trees_lab.py`. This lab uses the iris dataset that ships with scikit-learn, so there is no download.

---

## Step 1: Load and explore

**Why:** Iris has four features and three balanced classes. Before fitting anything, check what you are working with — class names, feature names, and how many samples each class has.

```python
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X, y = data.data, data.target

print("rows, features:", X.shape)
print("classes       :", data.target_names)
print("features      :", data.feature_names)
print("samples/class :", np.bincount(y))
```

**What to notice:** All three classes have 50 samples each, so this dataset is balanced. Imbalance is not a problem here, which lets you focus on the tree itself.

## Step 2: Split

**Why:** Hold back a test set before you fit anything. You judge a model on data it has not seen. `stratify=y` keeps the class proportions equal in both splits.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape)
```

---

## Step 3: Fit a full tree and witness the overfit

**Why:** A tree with no depth limit grows until every leaf is pure — it memorizes the training data. Training accuracy will be perfect, but test accuracy will be lower. This is the overfit you will fix in the next step.

```python
from sklearn.tree import DecisionTreeClassifier

full_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

print("full tree depth:", full_tree.get_depth())
print("train accuracy :", round(full_tree.score(X_train, y_train), 3))
print("test  accuracy :", round(full_tree.score(X_test,  y_test),  3))
```

**What to notice:** Training accuracy is 1.000 — the tree memorized every example. Test accuracy is lower, meaning it has not learned a rule that generalizes. The gap between the two numbers is the overfit.

> [!NOTE]
> **Checkpoint:** You can name the gap between train and test accuracy, and explain why a model that scores 1.0 on training data is not necessarily good.

## Step 4: Limit the depth and fix the overfit

**Why:** `max_depth` is the main lever against overfitting in decision trees. Shallower trees make coarser splits that generalize better. Compare train and test accuracy before and after to confirm the fix worked.

```python
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

print("depth-3 tree depth:", tree.get_depth())
print("train accuracy    :", round(tree.score(X_train, y_train), 3))
print("test  accuracy    :", round(tree.score(X_test,  y_test),  3))
```

> [!TIP]
> Try `max_depth` values of 1, 2, 3, and 4. Watch where train and test accuracy stop improving together — that is where the tree stops learning general patterns and starts memorizing noise.

---

## Step 5: Visualize the tree and read its decisions

**Why:** A decision tree is the rare model a domain expert can audit. Each internal node asks a yes/no question about one feature; follow the branches to a leaf and you have a plain-language rule. The color shows which class dominates each node.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
plot_tree(tree,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision tree (max depth 3)")
plt.tight_layout()
plt.show()
```

**How to read the tree:** Each box shows four things: the splitting question, the Gini impurity, the sample count, and the class distribution. Start at the root and follow the left branch when the condition is true, right when false. The leaf you land on is the prediction.

> [!NOTE]
> **Checkpoint:** Trace the path for a flower with petal length 1.2 cm. Which leaf do you reach, and which species does the tree predict?

## Step 6: Understand Gini impurity

**Why:** The tree chooses every split by finding the question that reduces Gini impurity the most. Understanding what Gini measures tells you how the model actually makes decisions.

```python
root_feature   = data.feature_names[tree.tree_.feature[0]]
root_threshold = tree.tree_.threshold[0]
root_gini      = tree.tree_.impurity[0]

print(f"Root splits on : {root_feature} <= {root_threshold:.2f}")
print(f"Root Gini      : {root_gini:.3f}")
print()
print("Gini = 0.0  →  pure node (one class only)")
print("Gini = 0.5  →  perfectly mixed (two classes, 50/50)")
print("The tree always picks the split that drops Gini the most.")
```

**What to notice:** Gini impurity is visible in the visualization too — dark nodes are nearly pure (low Gini), pale nodes are mixed. The tree works by turning pale nodes into dark ones, one split at a time.

---

## Step 7: Plot feature importance

**Why:** After fitting, the tree reports how much each feature contributed to reducing impurity across all splits. Features that never get used score zero. This tells you what the model actually learned to rely on.

```python
importances = tree.feature_importances_
order = np.argsort(importances)[::-1]

print("feature importance ranking:")
for i in order:
    bar = "#" * int(importances[i] * 40)
    print(f"  {data.feature_names[i]:<26} {importances[i]:.3f}  {bar}")

plt.figure(figsize=(7, 4))
plt.bar(range(len(importances)), importances[order])
plt.xticks(range(len(importances)),
           [data.feature_names[i] for i in order], rotation=20, ha="right")
plt.ylabel("importance")
plt.title("Feature importance")
plt.tight_layout()
plt.show()
```

**What to notice:** Petal measurements dominate; sepal width contributes almost nothing. A feature with near-zero importance is a candidate to remove.

> [!NOTE]
> **Checkpoint:** You can name the most and least important features, and explain what "importance" means in terms of Gini reduction across all splits.

## Step 8: Remove weak features and refit

**Why:** Dropping features that do not contribute simplifies the model without hurting accuracy. Simpler models are easier to explain and less likely to pick up noise. Test whether the leaner tree holds up on the same metric.

```python
# petal length = index 2, petal width = index 3
X_train_lean = X_train[:, 2:]
X_test_lean  = X_test[:,  2:]

lean_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train_lean, y_train)

print("lean tree (petal only) test accuracy:", round(lean_tree.score(X_test_lean, y_test), 3))
print("full tree (all features) test accuracy:", round(tree.score(X_test, y_test), 3))
```

> [!TIP]
> If the lean tree matches the full tree, you lost nothing by dropping the weak features. If accuracy drops significantly, the "weak" features were still carrying signal — keep them.

---

## Step 9: Judge it with a confusion matrix

**Why:** Accuracy is a single number; the confusion matrix shows exactly which species the model confuses with which. On a three-class problem, this tells you far more than a single score.

```python
from sklearn.metrics import confusion_matrix, classification_report

pred = tree.predict(X_test)

print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred, target_names=data.target_names))
```

**How to read the confusion matrix:** Rows are the true class, columns are the predicted class. Numbers on the diagonal are correct; off-diagonal numbers are mistakes. Look for which pairs of species the model struggles to separate.

> [!NOTE]
> **Checkpoint:** You can read the confusion matrix and say which species (if any) the model confuses, and which feature measurement likely causes the confusion.

---

## On your own

You are reviewing this classifier before it goes into a field identification app:

1. Try `max_depth` values of 2, 3, and 4 and report test accuracy for each. Which depth would you choose and why?
2. Visualize the lean tree (petal features only) alongside the full tree. Are the rules easier to explain with fewer features?
3. Write two sentences: what would you tell a botanist about which measurements actually matter, based on the feature importance chart?

## Responsible AI

> [!IMPORTANT]
> - A model trained on one population can fail on another. Iris is a clean textbook dataset; real-world data is messier, and the rules the tree learns may not transfer.
> - Explainability is not the same as correctness. A tree you can read is still wrong if it was trained on biased or unrepresentative data.
> - Before putting any classifier into a tool used by people, check its errors: which class does it misclassify most, and who bears the cost of those mistakes?

## What you learned

- Fit an unconstrained tree, recognize the overfit, and fix it with `max_depth`
- Visualize a decision tree and trace its rules from root to leaf
- Interpret Gini impurity as the splitting criterion the tree optimizes
- Rank features by importance and drop the ones that do not contribute
- Judge a multi-class classifier with a confusion matrix and classification report

## Stretch goals

- Use `cross_val_score` with `scoring="accuracy"` across `max_depth` values 1 through 8 and plot the results. Where does test accuracy peak before it starts to drop?
- Export the tree as text with `export_text` from `sklearn.tree` and compare it to the visual — which is easier to audit with a domain expert?
- Try `criterion="entropy"` instead of the default Gini and see if it changes the tree structure or test accuracy.
