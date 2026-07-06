# Lab 3: A House-Price Estimator with Linear Regression

> **Machine Learning with AI and Python** · Day 1
> Prerequisite: Lab 2 complete

## The scenario

A real-estate team wants a price estimator they can trust and explain to a client. "Trust" and "explain" are the hard parts. You will build the model end to end: explore the data, fit a baseline, validate it honestly, interpret it, diagnose where it fails, try to improve it, and package it so someone else can use it. This is the full arc of a small regression project.

## Why this lab matters

The slides covered fitting a line, reading MAE, RMSE, and R squared, and checking for overfitting. Real projects add a few habits that separate a demo from something you would ship: cross-validation (so a lucky split does not fool you), residual analysis (so you see *where* it fails), regularization (so extra features do not cause overfitting), and saving the model. You will practice all of them.

## What you will do

- Explore the data and pick features
- Fit a baseline, then validate it with cross-validation
- Interpret coefficients and diagnose with a residual plot
- Improve with more features, scaling, polynomial terms, and regularization
- Predict a new house and save the model for reuse

## Before you start

You will use Python with scikit-learn, pandas, and matplotlib. Create `regression_lab.py`. The first run of `fetch_california_housing` downloads once; if you are offline, swap in `load_diabetes(as_frame=True)` as in Lab 2 (the workflow is identical).

---

## Step 1: Load and explore

**Why:** Never model data you have not looked at. A quick profile and a correlation check tell you which features are likely to matter and whether anything is off.

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.shape)
print(df.describe().round(2))
print(df.corr()["MedHouseVal"].sort_values(ascending=False).round(2))
```

**What to notice:** `MedInc` (median income) correlates most with the target `MedHouseVal` (median home value, in $100,000s). That is your strongest single predictor.

### Step 2: Pick features and split

**Why:** Start with a few features you can reason about, so the model stays explainable. Hold back a test set, because you judge a model on data it has not seen. `random_state` makes the split repeatable.

```python
from sklearn.model_selection import train_test_split

X = df[["MedInc", "AveRooms", "HouseAge"]]
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
```

---

## Step 3: Fit a baseline and evaluate

**Why:** A simple baseline is your reference point. Every later change has to beat it, or it is not worth the complexity.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(root_mean_squared_error(y_test, pred), 3))
print("R2  :", round(r2_score(y_test, pred), 3))
```

**How to read these:** MAE is the average miss in the target's units (an MAE of `0.5` is about $50,000). RMSE punishes big misses more. R squared is the share of variation explained, from 0 to 1.

> [!NOTE]
> **Checkpoint:** You can state the baseline's average error in dollars and roughly what fraction of the variation it explains.

## Step 4: Validate with cross-validation

**Why:** A single train/test split can be lucky or unlucky. Cross-validation trains and tests on several different splits and reports the spread, which is a far more honest estimate of real-world performance.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("fold R2 scores:", scores.round(3))
print("mean R2:", round(scores.mean(), 3), "+/-", round(scores.std(), 3))
```

> [!WARNING]
> If your single-split R squared is much higher than the cross-validation mean, that split flattered the model. Trust the cross-validation number when you report performance.

## Step 5: Interpret the model

**Why:** The team needs to explain a price to a client. A linear model is easy to interpret: each feature has a weight, and the sign and size say how the model uses it.

```python
for name, weight in zip(X.columns, model.coef_):
    print(name, round(weight, 3))
print("intercept:", round(model.intercept_, 3))
```

**What to notice:** A positive weight means the prediction rises as that feature rises. Do not read a weight as proof of cause; correlation is not causation.

## Step 6: Diagnose with a residual plot

**Why:** Metrics summarize; a residual plot shows *where* the model fails. Residuals (actual minus predicted) should look like random noise around zero. A pattern (a curve, or a fan that widens) means a straight line was the wrong shape.

```python
import matplotlib.pyplot as plt

residuals = y_test - pred
plt.scatter(pred, residuals, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted"); plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals should look like random noise")
plt.show()
```

> [!NOTE]
> **Checkpoint:** You looked at the residuals and can say whether they look like noise or show a pattern. On this dataset you will see the model struggles at the high end (prices are capped).

---

## Step 7: Improve, attempt 1, use all features

**Why:** More features can help. The honest test is the held-out score, not the training fit.

```python
X_all = data.data                       # all eight features
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    X_all, y, test_size=0.2, random_state=42)
full = LinearRegression().fit(Xa_tr, ya_tr)

print("3-feature test R2:", round(model.score(X_test, y_test), 3))
print("8-feature test R2:", round(full.score(Xa_te, ya_te), 3))
```

## Step 8: Improve, attempt 2, scaling and polynomial features

**Why:** Some relationships are curved. Polynomial features let a linear model bend, and a `Pipeline` keeps the steps together so scaling is learned from training data only. This previews how real models are assembled.

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

poly = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    LinearRegression()
).fit(Xa_tr, ya_tr)
print("polynomial test R2:", round(poly.score(Xa_te, ya_te), 3))
```

> [!WARNING]
> Polynomial features can overfit fast: the training score jumps but the test score may not. Always compare on the test set, and watch the gap.

## Step 9: Improve, attempt 3, regularization

**Why:** With many (or polynomial) features, regularization penalizes large weights so the model generalizes instead of memorizing. `RidgeCV` even picks the penalty strength for you with built-in cross-validation.

```python
from sklearn.linear_model import Ridge, Lasso, RidgeCV

ridge = Ridge(alpha=1.0).fit(Xa_tr, ya_tr)
lasso = Lasso(alpha=0.1).fit(Xa_tr, ya_tr)
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(Xa_tr, ya_tr)

print("ridge  test R2:", round(ridge.score(Xa_te, ya_te), 3))
print("lasso  test R2:", round(lasso.score(Xa_te, ya_te), 3))
print("lasso kept:", int((lasso.coef_ != 0).sum()), "of", len(lasso.coef_), "features")
print("RidgeCV chose alpha:", ridge_cv.alpha_)
```

**What to notice:** Ridge (L2) shrinks all weights toward zero; Lasso (L1) can set some to exactly zero, which selects features for you. `alpha` is the dial: higher means simpler.

> [!NOTE]
> **Checkpoint:** You have several models and their test R squared side by side, and can say which one you would ship and why.

---

## Step 10: Use the model

**Why:** A model earns its keep when you can score a new example, the kind of thing behind a quote tool. Pass a DataFrame with the same columns you trained on.

```python
new = pd.DataFrame([{"MedInc": 5.0, "AveRooms": 6, "HouseAge": 15}])
value = model.predict(new)[0]
print(f"predicted median value: ${value * 100000:,.0f}")
```

## Step 11: Save it for reuse

**Why:** You do not retrain every time you predict. Save the fitted model to disk, then load it wherever predictions are needed. `joblib` is the standard tool.

```python
import joblib

joblib.dump(model, "house_model.joblib")
loaded = joblib.load("house_model.joblib")
print("loaded model test R2:", round(loaded.score(X_test, y_test), 3))
```

> [!NOTE]
> **Checkpoint:** You saved and reloaded a model and got the same test score, so the saved artifact works.

## On your own

You are the analyst on this project. Produce the best model you honestly can and justify it:

1. Choose a feature set (start from the correlation ranking in Step 1).
2. Compare at least three approaches (for example plain, polynomial, and Ridge) using `cross_val_score`, not a single split.
3. Report the mean cross-validation R squared for each, and say which you would ship and why.

Use Copilot to draft the code, then verify every score yourself.

## Responsible AI

> [!IMPORTANT]
> - A price model trained on one place or era can be badly wrong elsewhere. Do not predict far outside the range of your training data.
> - Correlation is not causation. A coefficient does not mean changing that feature would change the price.
> - Be able to explain any number the model produces before it drives a decision that affects a person.

## What you learned

- The load, split, fit, predict, evaluate workflow, and why cross-validation beats a single split
- How to read MAE, RMSE, and R squared, and interpret coefficients
- How to diagnose with a residual plot and recognize overfitting
- How to improve with more features, scaling, polynomial terms, and regularization, and how to save a model

## Stretch goals

- Use `GridSearchCV` to tune `alpha` for Ridge across a wider range, and compare to `RidgeCV`.
- Add the `rooms_per_person` feature idea from Lab 2 and test whether cross-validation R squared improves.
- Plot predicted versus actual and mark the points where the model is worst. What do they have in common?
