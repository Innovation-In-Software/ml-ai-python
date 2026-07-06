# Lab 4: A First Neural Network

> **Machine Learning with AI and Python** · Day 2
> Prerequisite: Lab 3 (Day 2) complete

## The scenario

A postal service wants to read handwritten digits on envelopes automatically. Each digit is a tiny 8×8 grayscale image — 64 pixel values — and the model must classify it as 0 through 9. This is the task that made neural networks famous. Today you will build one, see it learn in real time, and understand why it works on image-like data where simpler models start to struggle.

## Why this lab matters

Every model you have built so far — linear regression, logistic regression, decision trees, random forests — makes predictions through a single transformation of the input features. A neural network stacks many transformations, each one learning to represent the data at a higher level of abstraction. That depth is what lets it find patterns that flat models miss. But depth also brings new risks: slower training, more ways to overfit, and harder interpretability. This lab shows you the power, the pitfall, and the fix — in that order.

## What you will do

- Load and visualize digit images to understand the input
- Scale features and set a logistic regression baseline
- Build a neural network, understand its architecture, and beat the baseline
- Experiment with depth to see when more layers help and when they do not
- Plot the training loss curve to watch the model learn
- Introduce overfitting deliberately, then fix it with early stopping
- Read a confusion matrix to see which digits the model still confuses

## Before you start

You will use Python with scikit-learn and matplotlib. Create `neural_network_lab.py`. This lab uses the digits dataset that ships with scikit-learn, so there is no download.

---

## Step 1: Load and visualize the digits

**Why:** The digits dataset contains 1797 handwritten digit images, each stored as 64 pixel values (an 8×8 grid flattened into a row). Visualizing a few images first makes the problem concrete — you are asking a model to read handwriting, which is exactly the kind of task neural networks were built for.

```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

data = load_digits()
X, y = data.data, data.target

print("images, pixels:", X.shape)    # (1797, 64)
print("classes:", np.unique(y))      # 0 through 9
print("pixel value range:", X.min(), "to", X.max())

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, img, label in zip(axes.flat, data.images, data.target):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(str(label))
    ax.axis("off")
plt.suptitle("Sample digit images (8×8 pixels each)")
plt.tight_layout()
plt.show()
```

**What to notice:** Each image is tiny — 8×8 pixels. Yet humans read these instantly. The question is how much structure a model needs to do the same.

## Step 2: Scale and split

**Why:** Pixel values run from 0 to 16. Scaling brings them to a common range, which speeds up gradient-based training and prevents features with larger raw values from dominating. You saw the same requirement in Lab 3 for distance-based clustering — the underlying reason is the same here.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape)
```

---

## Step 3: Set a baseline with logistic regression

**Why:** Always know what a simpler model can do before adding complexity. Logistic regression is the model you know best from Day 1. If the neural network cannot beat it, the extra complexity is not justified.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"logistic regression accuracy: {lr_acc:.3f}")
```

> [!NOTE]
> **Checkpoint:** Record this number. Every model you build in the rest of the lab is compared against it.

## Step 4: Build a neural network and understand the architecture

**Why:** A neural network is a sequence of layers. Each layer takes its inputs, multiplies them by learned weights, adds a bias, and passes the result through an activation function. The activation (ReLU here) introduces non-linearity — without it, stacking layers would be mathematically equivalent to one layer. `hidden_layer_sizes=(64,)` means one hidden layer with 64 neurons.

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(64,),   # one hidden layer, 64 neurons
    activation="relu",          # ReLU: output = max(0, input)
    max_iter=500,
    random_state=42
).fit(X_train, y_train)

mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
print(f"neural network accuracy:      {mlp_acc:.3f}")
print(f"logistic regression accuracy: {lr_acc:.3f}")
print(f"improvement: {mlp_acc - lr_acc:+.3f}")
```

**How the architecture works:**
- **Input layer:** 64 neurons — one per pixel
- **Hidden layer:** 64 neurons, each connected to all inputs, each applying ReLU
- **Output layer:** 10 neurons — one per digit class; the highest score wins

> [!NOTE]
> **Checkpoint:** The neural network should outperform logistic regression. If the gap is small on this dataset, that is informative too — not every problem needs depth.

---

## Step 5: Experiment with depth

**Why:** More layers let the network learn more abstract representations. But more layers also mean more parameters to tune, slower training, and higher risk of overfitting. Test three architectures and see where depth stops helping.

```python
architectures = [
    (64,),
    (64, 32),
    (128, 64, 32),
]

print(f"{'architecture':<20} {'test accuracy':>14}")
print("-" * 36)
for layers in architectures:
    m = MLPClassifier(hidden_layer_sizes=layers, activation="relu",
                      max_iter=500, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"{str(layers):<20} {acc:>14.3f}")
```

**What to notice:** Accuracy may improve from one to two layers, then plateau or drop. Adding layers is not free — the model has more to learn and can start memorising instead of generalising.

> [!TIP]
> In practice, architecture search is often done with cross-validation rather than a single test split. What you see here is a quick signal, not a final answer.

## Step 6: Plot the training loss curve

**Why:** Unlike most sklearn models, a neural network trains iteratively — it makes many passes over the training data, adjusting weights a little each time. The loss curve shows that process: loss should fall steeply at first, then level off. Watching it helps you understand whether training ran long enough and whether the model is still improving.

```python
mlp_long = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
).fit(X_train, y_train)

plt.figure(figsize=(7, 4))
plt.plot(mlp_long.loss_curve_)
plt.xlabel("training iteration")
plt.ylabel("loss")
plt.title("Training loss — falling loss means the model is learning")
plt.tight_layout()
plt.show()

print("iterations run:", mlp_long.n_iter_)
```

**What to notice:** If the curve is still falling sharply at the end, `max_iter` may be too low. If it flattened out early, the model converged before the limit — good.

---

## Step 7: Introduce overfitting, then fix it with early stopping

**Why:** A very large network will eventually memorise the training data — training accuracy reaches near 100% while test accuracy stalls or drops. Early stopping monitors a held-out validation set during training and stops when performance there stops improving, the same principle as limiting `max_depth` in Lab 1.

```python
# Overfit deliberately with a very large network and no stopping
overfit = MLPClassifier(
    hidden_layer_sizes=(256, 256, 256),
    activation="relu",
    max_iter=1000,
    random_state=42
).fit(X_train, y_train)

print("overfit model:")
print("  train accuracy:", round(accuracy_score(y_train, overfit.predict(X_train)), 3))
print("  test  accuracy:", round(accuracy_score(y_test,  overfit.predict(X_test)),  3))

# Fix with early stopping
stopped = MLPClassifier(
    hidden_layer_sizes=(256, 256, 256),
    activation="relu",
    max_iter=1000,
    early_stopping=True,     # holds out 10% of training data to monitor
    random_state=42
).fit(X_train, y_train)

print("\nearly stopping model:")
print("  train accuracy:", round(accuracy_score(y_train, stopped.predict(X_train)), 3))
print("  test  accuracy:", round(accuracy_score(y_test,  stopped.predict(X_test)),  3))
print("  stopped at iteration:", stopped.best_loss_)
```

> [!WARNING]
> A high train accuracy paired with a meaningfully lower test accuracy is always a sign of overfitting — regardless of how good the train number looks. The test score is the only one that matters for deployment.

## Step 8: Confusion matrix — which digits still get confused?

**Why:** Accuracy is a single number; the confusion matrix shows exactly where the model makes mistakes. On a 10-class problem this is far more informative — you can see which digit pairs look similar to the model and where human-level difficulty shows up.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

best_mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
).fit(X_train, y_train)

pred = best_mlp.predict(X_test)

print(classification_report(y_test, pred))

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred),
    display_labels=data.target_names
)
fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix — neural network on digits")
plt.tight_layout()
plt.show()
```

**What to notice:** Mistakes tend to cluster around visually similar pairs: 3 and 8, 4 and 9, 1 and 7. These are the same pairs humans sometimes hesitate on. Off-diagonal values represent errors; a nearly diagonal matrix means strong performance.

> [!NOTE]
> **Checkpoint:** You can name two digit pairs the model most often confuses and explain why those specific pairs make sense visually.

---

## On your own

You are evaluating this model for a postal sorting system:

1. The system processes millions of envelopes per day. A 97% accurate model still misreads thousands of digits — for which digits (based on the confusion matrix) would you want a human to double-check the model's output?
2. Try `activation="tanh"` instead of `"relu"` on the (64, 32) architecture. Does it improve or hurt accuracy? Look up what tanh does differently from ReLU and write one sentence explaining the trade-off.
3. Compare the neural network to a `RandomForestClassifier` on the same digits data. Which wins on accuracy? Which would you trust more for this task and why?

## Responsible AI

> [!IMPORTANT]
> - Neural networks are harder to explain than decision trees or logistic regression. Before deploying one in a consequential system, ask whether the accuracy gain is worth the loss of interpretability.
> - A model trained on one population's handwriting can fail on another's. Digits written in different cultures or by people with motor impairments may fall outside the training distribution.
> - High overall accuracy can hide poor performance on a specific class. Always check the per-class rows of the classification report before declaring the model ready.

## What you learned

- A neural network stacks layers of weighted connections with non-linear activations between them
- Scaling is required before training because gradient descent is sensitive to feature magnitude
- Depth (more layers) can improve performance but also increases overfitting risk
- The training loss curve shows the model learning iteratively — watch for convergence, not just final loss
- Early stopping is the neural network equivalent of `max_depth`: it prevents memorisation by stopping training when validation performance peaks
- Confusion matrices on multi-class problems reveal which specific errors the model makes, not just how many

## Stretch goals

- Use `GridSearchCV` to search over `hidden_layer_sizes` and `alpha` (the regularisation strength) and find the best architecture by cross-validated accuracy.
- Try `MLPClassifier` with `solver="sgd"` and `learning_rate_init=0.01`. Plot the loss curve alongside the default `solver="adam"` and compare convergence speed.
- Visualize what the first hidden layer has learned: reshape the weight matrix `best_mlp.coefs_[0]` into 64 images of 8×8 pixels and plot them — each image is a "feature detector" the network learned.
