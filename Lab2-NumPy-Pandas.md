# Lab 2: Data Wrangling with NumPy and Pandas

> **Machine Learning with AI and Python** · Day 1
> Prerequisite: Lab 1 complete

## The scenario

A colleague exports `homes.csv` from three different systems and drops it on you. It has duplicate rows, missing values, inconsistent text ("Reno", "RENO", "reno "), numbers stored as text, broken dates, and at least one obvious data-entry error. Your job: turn this mess into analysis-ready data and prepare features for a model. This is what real data work looks like, and it is where most ML time actually goes.

## Why this lab matters

The slides covered NumPy arrays, Pandas DataFrames, and the "table to model input" flow. Real datasets rarely arrive clean, so this lab practices the full cleaning pipeline: profiling, fixing types, normalizing text, handling missing values, removing duplicates and outliers, then summarizing, encoding, scaling, and splitting into `X` and `y`. Use Copilot as you go: when a method is new, select it and run `/explain`.

## What you will do

- Part A: NumPy foundations (vectorization, broadcasting, boolean masks, axes)
- Part B: Create and profile a messy dataset
- Part C: Clean it (types, text, missing values, duplicates, outliers)
- Part D: Explore it (group, pivot, correlate)
- Part E: Prepare features (engineer, encode, scale, split, save)

## Before you start

You will use Python with NumPy, Pandas, and scikit-learn. Create `data_lab.py`, or a Jupyter notebook if you prefer to run one cell at a time.

---

## Part A: NumPy foundations

### Step 1: Arrays, vectorization, and shape

**Why:** The NumPy array is the structure the whole ML ecosystem stands on. Vectorized math (no loops) is why it is fast, and `shape`/`dtype` are the first things to check when something breaks.

```python
import numpy as np

sqft  = np.array([1800, 1200, 2400, 1500])
price = np.array([450000, 300000, 650000, 410000])

print(sqft.shape, sqft.dtype)      # (4,) int64
price_per_sqft = price / sqft      # vectorized, elementwise
print(price_per_sqft.round(0))     # [250. 250. 271. 273.]
```

### Step 2: Two dimensions, axes, and broadcasting

**Why:** Real data is 2D. Summarizing along an axis and broadcasting a row across a grid are moves you will use constantly, and `axis` is the number-one beginner confusion.

```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(grid.mean(axis=0))           # column means: [2.5 3.5 4.5]
print(grid.sum(axis=1))            # row sums:     [ 6 15]
print(grid + np.array([10, 20, 30]))   # broadcast a row across both rows
```

### Step 3: Boolean masks

**Why:** Selecting the rows that matter (over a size, above a price) is a daily task, and masks do it without a loop. This is the NumPy idea under Pandas filtering.

```python
print(sqft[sqft > 1500])                      # [1800 2400]
print(np.where(price > 400000, "high", "low"))
```

> [!NOTE]
> **Checkpoint:** You can explain `axis=0` versus `axis=1`, and keep only the elements that meet a condition.

---

## Part B: Create and profile a messy dataset

### Step 4: Generate the raw file

**Why:** Using a messy file (not a tidy one) is the whole point. Run this once to create `homes.csv`. Notice the deliberate problems: a duplicate row, inconsistent city text, a missing size, a bed count stored as text, prices as strings, mixed date formats, and a `999999` size that is clearly a typo.

```python
import pandas as pd

raw = pd.DataFrame({
    "city":   ["Austin", "austin ", "Reno", "RENO", "Miami", None, "Boise", "Reno"],
    "sqft":   [1800, 1800, 1200, 2400, None, 1500, 2000, 999999],
    "beds":   [3, 3, 2, 4, 3, 3, "4", 3],
    "price":  ["450000", "450000", "300000", "650000", "410000", "310000", "520000", "305000"],
    "listed": ["2026-01-05", "2026-01-05", "2026/02/11", "2026-03-01",
               "n/a", "2026-04-20", "2026-05-02", "2026-02-15"],
})
raw.to_csv("homes.csv", index=False)
print("wrote homes.csv")
```

### Step 5: Load and profile

**Why:** Before you fix anything, look. `info()`, `describe()`, `isna().sum()`, and `duplicated()` tell you what is wrong and where.

```python
df = pd.read_csv("homes.csv")
print(df.info())                 # dtypes + non-null counts; note beds and price are objects (text)
print(df.isna().sum())           # missing values per column
print("duplicate rows:", df.duplicated().sum())
print(df["city"].unique())       # inconsistent capitalization and spacing
```

> [!NOTE]
> **Checkpoint:** You can name at least four problems in this file from the profile alone.

---

## Part C: Clean the data

### Step 6: Normalize text and drop duplicates

**Why:** "Reno", "RENO", and "reno " should be one city. Normalize first, then remove duplicates, so near-duplicates collapse correctly.

```python
df["city"] = df["city"].str.strip().str.title()   # "austin " -> "Austin"; None stays NaN
df = df.drop_duplicates()
print(df["city"].unique())
print("rows after dedup:", len(df))
```

### Step 7: Fix data types

**Why:** `beds` and `price` came in as text, and `listed` as strings. Models need numbers and real dates. `errors="coerce"` turns anything unparseable into a missing value instead of crashing.

```python
df["beds"]   = pd.to_numeric(df["beds"], errors="coerce")
df["price"]  = pd.to_numeric(df["price"], errors="coerce")
df["listed"] = pd.to_datetime(df["listed"], errors="coerce", format="mixed")
print(df.dtypes)
```

### Step 8: Handle missing values on purpose

**Why:** A missing value will break most models. Decide per column: drop rows you cannot use, fill the ones you can. Dropping a row with no city or price is honest; filling a missing size with the median is reasonable.

```python
df = df.dropna(subset=["city", "price"])          # cannot model these rows
df["sqft"] = df["sqft"].fillna(df["sqft"].median())
print(df.isna().sum())
```

### Step 9: Find and handle outliers

**Why:** A `999999` square-foot home is a data-entry error that will drag any model. The IQR rule is a standard, defensible way to flag extreme values.

```python
q1, q3 = df["sqft"].quantile([0.25, 0.75])
iqr = q3 - q1
high = q3 + 1.5 * iqr
print("upper fence:", high)
before = len(df)
df = df[df["sqft"] <= high]
print(f"removed {before - len(df)} outlier row(s)")
```

> [!WARNING]
> Do not delete outliers reflexively. Some are real (a genuine mansion). Investigate first; here `999999` is clearly a typo, so removing it is justified.

> [!NOTE]
> **Checkpoint:** `df.dtypes` shows numeric `beds`, `price`, and a real datetime `listed`, and there are no missing values or obvious outliers left.

---

## Part D: Explore the clean data

### Step 10: Summarize by group and pivot

**Why:** Real questions are about groups. `groupby`, `value_counts`, and `pivot_table` answer "average price per city" and "how do price and size relate."

```python
print(df["city"].value_counts())
print(df.groupby("city")["price"].mean().round(0))
print(df.sort_values("price", ascending=False).head(3)[["city", "price"]])
```

### Step 11: Correlate numeric features

**Why:** Correlation is a quick read on which features move with price, which guides feature choice for the next lab.

```python
print(df[["sqft", "beds", "price"]].corr()["price"].round(2))
```

> [!TIP]
> Ask Copilot Chat to `/explain` `df.groupby("city")["price"].mean()`. Then confirm the split-apply-combine description against the output you see.

---

## Part E: Prepare features for a model

### Step 12: Engineer a feature

**Why:** A derived feature can capture something raw columns do not, like price intensity. Feature engineering is where domain knowledge meets code.

```python
df["price_per_sqft"] = (df["price"] / df["sqft"]).round(0)
print(df[["city", "sqft", "price", "price_per_sqft"]])
```

### Step 13: Encode categories

**Why:** Models do math, so text like "Austin" must become numbers. One-hot encoding makes one 0/1 column per city, which does not imply a false ordering the way labeling them 1, 2, 3 would.

```python
encoded = pd.get_dummies(df, columns=["city"])
print(encoded.columns.tolist())
```

### Step 14: Split into X and y, then scale

**Why:** Every model wants features apart from the target. Scaling puts features on the same footing for the models that need it, and fitting the scaler on training data only avoids leaking test information.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_cols = [c for c in encoded.columns if c not in ("price", "listed")]
X = encoded[feature_cols]
y = encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

scaler = StandardScaler().fit(X_train)       # fit on train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(X_train.shape, X_test.shape)
```

### Step 15: Save the cleaned data

**Why:** Clean data is an asset. Save it so the next step (and the next lab) starts from tidy data, not the mess.

```python
df.to_csv("homes_clean.csv", index=False)
print("wrote homes_clean.csv")
```

> [!NOTE]
> **Checkpoint:** You produced `homes_clean.csv`, an encoded feature matrix `X`, a target `y`, and a train/test split with the scaler fit on training data only.

---

## Bonus: profile a real dataset

**Why:** The same moves scale to real data. The California housing dataset ships with scikit-learn and is a good profiling target.

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True).frame
print(housing.shape)
print(housing.describe().round(2))
print(housing.corr()["MedHouseVal"].sort_values(ascending=False).round(2))
```

> [!NOTE]
> `fetch_california_housing` downloads once and caches locally, so you need internet the first time. If you are offline, use `from sklearn.datasets import load_diabetes` and `load_diabetes(as_frame=True)`. The steps are identical.

## On your own

Starting from `homes_clean.csv`, answer three questions with Pandas and check your code with Copilot's `/explain`:

- Which city has the highest average `price_per_sqft`?
- How many homes were listed after March 1, 2026? (Hint: compare the `listed` datetime column.)
- What is the correlation between `beds` and `price`?

Write one sentence on which feature you would trust most to predict price, and why.

## Responsible AI

> [!IMPORTANT]
> Real datasets can contain personal or sensitive data. Know where your data comes from and what is in it before you load it, and keep private data on systems you control. Cleaning choices (which rows you drop, which values you fill) change the story the data tells, so document them.

## What you learned

- NumPy arrays give you fast, vectorized math, broadcasting, and boolean filtering
- Profiling with `info`, `describe`, `isna`, and `duplicated` before you change anything
- A full cleaning pipeline: normalize text, fix types with `coerce`, handle missing values and outliers on purpose
- Preparing features: engineering, one-hot encoding, scaling on training data only, and splitting into `X` and `y`

## Stretch goals

- Replace the median fill for `sqft` with a per-city median (`groupby("city")["sqft"].transform("median")`). Does it change the outcome?
- Bucket `price` into low, medium, and high with `pd.qcut`, then compare average `sqft` per bucket.
- Use `pd.pivot_table` to show average `price_per_sqft` by city and bed count.
