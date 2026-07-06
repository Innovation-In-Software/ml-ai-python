# Lab 4: A First Neural Network with Keras

> **Machine Learning with AI and Python** · Day 2
> Prerequisite: Lab 3 (Day 2) complete

## The scenario

A postal service wants to read handwritten digits on envelopes automatically. Each digit is a tiny 8×8 grayscale image — 64 pixel values — and the model must classify it as 0 through 9. This is the task that made neural networks famous. Today you will build one with Keras, watch it learn epoch by epoch, and understand why depth helps on image-like data where simpler models start to struggle.

## Why this lab matters

Every model you have built so far makes predictions through a single transformation of the input features. A neural network stacks many transformations — called layers — each one learning to represent the data at a higher level of abstraction. Keras is the library that makes building those layers readable: you assemble a network the way you would describe it on a whiteboard. That depth is what lets it find patterns flat models miss, but it also brings new risks. This lab shows you the power, the pitfall, and the fix — in that order.

## What you will do

- Load and visualize digit images to understand the input
- Scale features and set a logistic regression baseline
- Build a Keras neural network, read its summary, and beat the baseline
- Plot training history to watch the model learn epoch by epoch
- Experiment with depth to see when more layers help and when they do not
- Introduce overfitting deliberately and fix it with an early stopping callback
- Read a confusion matrix to see which digits the model still confuses

## Before you start

You will use Python with scikit-learn, TensorFlow/Keras, and matplotlib. Create `neural_network_lab.py`.

Install TensorFlow if you have not already:

```
pip install tensorflow
```

This lab uses the digits dataset that ships with scikit-learn, so there is no data download.

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

**What to notice:** Each image is tiny — 8×8 pixels. The 64 pixel values are the features your network will learn from.

## Step 2: Scale and split

**Why:** Pixel values run from 0 to 16. Scaling brings them close to zero with unit variance, which makes gradient-based training faster and more stable. You saw the same requirement in Lab 3 for distance-based clustering — the underlying reason is the same here.

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

**Why:** Always know what a simpler model can do before adding complexity. If the neural network cannot clearly beat logistic regression here, the extra machinery is not justified.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"logistic regression accuracy: {lr_acc:.3f}")
```

> [!NOTE]
> **Checkpoint:** Record this number. Every model you build in the rest of the lab is compared against it.

## Step 4: Build your first Keras model

**Why:** In Keras you describe a network by stacking layers. `Sequential` means data flows straight through from input to output. Each `Dense` layer is fully connected — every neuron receives input from every neuron in the previous layer. `relu` activation introduces non-linearity between layers; `softmax` on the output layer converts raw scores into probabilities that sum to 1, one per class.

```python
import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(64,)),              # 64 pixels in
    layers.Dense(64, activation="relu"),   # hidden layer: 64 neurons
    layers.Dense(10, activation="softmax") # output layer: one score per digit
])

model.summary()
```

**How to read the summary:** Each row is a layer. `Output Shape` shows the data dimensions after that layer. `Param #` is the number of weights the model will learn in that layer — the first Dense layer alone learns 64×64 + 64 = 4,160 numbers.

> [!NOTE]
> **Checkpoint:** Add up the total parameters from `model.summary()`. Every one of those numbers gets adjusted during training.

---

## Step 5: Compile, train, and compare to the baseline

**Why:** `compile` wires up the learning process: the optimizer adjusts weights, the loss measures how wrong the model is, and metrics are what you monitor. `fit` runs the training loop. `validation_split` holds back 10% of training data so you can watch train versus validation accuracy in real time — one line per epoch.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

nn_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\nneural network accuracy:      {nn_acc:.3f}")
print(f"logistic regression accuracy: {lr_acc:.3f}")
```

**What to notice:** Each epoch prints training loss, training accuracy, validation loss, and validation accuracy. Watch validation accuracy climb toward training accuracy — that is the model generalising, not just memorising.

> [!TIP]
> `sparse_categorical_crossentropy` works when your labels are integers (0–9). If you one-hot encoded them you would use `categorical_crossentropy` instead.

## Step 6: Plot the training history

**Why:** The `history` object records every metric for every epoch. Plotting train versus validation curves turns the numbers from Step 5 into a picture of the learning process — you can see exactly when the model converges and whether it is starting to overfit.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(history.history["accuracy"],     label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout()
plt.show()
```

**What to notice:** When training and validation curves stay close together, the model is generalising. When they diverge — training keeps improving but validation flattens or drops — that is overfitting beginning.

---

## Step 7: Experiment with depth

**Why:** More layers let the network learn more abstract patterns. But more layers also mean more parameters, slower training, and higher risk of overfitting. Test a few architectures side by side to see where depth stops helping.

```python
architectures = {
    "shallow  (64)":       [64],
    "medium   (64, 32)":   [64, 32],
    "deep  (128, 64, 32)": [128, 64, 32],
}

print(f"{'architecture':<26} {'test accuracy':>14}")
print("-" * 42)

for name, units in architectures.items():
    m = keras.Sequential(
        [keras.Input(shape=(64,))]
        + [layers.Dense(u, activation="relu") for u in units]
        + [layers.Dense(10, activation="softmax")]
    )
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    m.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_split=0.1, verbose=0)
    acc = m.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{name:<26} {acc:>14.3f}")
```

**What to notice:** Accuracy may improve from one to two layers, then plateau. Adding layers is not free — the model has more to learn and can start memorising instead of generalising.

## Step 8: Introduce overfitting, then fix it with early stopping

**Why:** A very large network trained too long will memorise the training data — training accuracy climbs toward 100% while validation accuracy peaks and then drops. An `EarlyStopping` callback watches validation loss and stops training automatically when it stops improving, the same principle as limiting `max_depth` in Lab 1.

```python
from keras.callbacks import EarlyStopping

# Overfit deliberately: large network, many epochs, no stopping
big_model = keras.Sequential([
    keras.Input(shape=(64,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10,  activation="softmax"),
])
big_model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
hist_overfit = big_model.fit(
    X_train, y_train, epochs=200,
    batch_size=32, validation_split=0.1, verbose=0
)

# Same network with early stopping
big_model_es = keras.Sequential([
    keras.Input(shape=(64,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10,  activation="softmax"),
])
big_model_es.compile(optimizer="adam",
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
hist_stopped = big_model_es.fit(
    X_train, y_train, epochs=200,
    batch_size=32, validation_split=0.1,
    callbacks=[early_stop], verbose=0
)

print(f"overfit model stopped at epoch: 200")
print(f"early stopping stopped at epoch: {early_stop.stopped_epoch + 1}")

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, hist, title in [
    (axes[0], hist_overfit,  "No early stopping"),
    (axes[1], hist_stopped,  "With early stopping"),
]:
    ax.plot(hist.history["accuracy"],     label="train")
    ax.plot(hist.history["val_accuracy"], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title(title); ax.legend()
plt.tight_layout()
plt.show()
```

**What to notice:** The left plot shows the classic overfit pattern — training accuracy climbs but validation diverges. The right plot stops at the best validation point. `restore_best_weights=True` means the saved model is from that best epoch, not the last one.

> [!WARNING]
> A high training accuracy paired with lower validation accuracy is always a sign of overfitting, regardless of how good the training number looks. The validation score is the only one that matters.

---

## Step 9: Confusion matrix — which digits still get confused?

**Why:** Accuracy is a single number; the confusion matrix shows exactly where the model makes mistakes. On a 10-class problem this reveals which specific digit pairs the model struggles with — information a single score completely hides.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

best_model = keras.Sequential([
    keras.Input(shape=(64,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
best_model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
best_model.fit(X_train, y_train, epochs=50, batch_size=32,
               validation_split=0.1, verbose=0)

pred_probs = best_model.predict(X_test)
pred = pred_probs.argmax(axis=1)

print(classification_report(y_test, pred))

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred),
    display_labels=data.target_names
)
fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix — Keras neural network on digits")
plt.tight_layout()
plt.show()
```

**What to notice:** Mistakes tend to cluster around visually similar pairs — 3 and 8, 4 and 9, 1 and 7. The same pairs humans sometimes hesitate on. A nearly diagonal matrix means strong performance across all classes.

> [!NOTE]
> **Checkpoint:** You can name two digit pairs the model most often confuses and explain why those specific pairs make sense visually.

---

## On your own

You are evaluating this model for a postal sorting system:

1. The system processes millions of envelopes per day. A 97% accurate model still misreads thousands of digits — for which digits (based on the confusion matrix) would you want a human to double-check the model's output?
2. Add a `layers.Dropout(0.3)` layer after each Dense hidden layer and retrain. Does dropout improve validation accuracy? Look up what dropout does and write one sentence explaining it as an overfitting prevention technique.
3. Compare the neural network to a `RandomForestClassifier` on the same digits data using the same train/test split. Which wins on accuracy, and which would you trust more in a production system and why?

## Responsible AI

> [!IMPORTANT]
> - Neural networks are harder to explain than decision trees or logistic regression. Before deploying one in a consequential system, ask whether the accuracy gain is worth the loss of interpretability.
> - A model trained on one population's handwriting can fail on another's. Digits written in different cultures or by people with motor impairments may fall outside the training distribution.
> - High overall accuracy can hide poor performance on a specific class. Always check the per-class rows of the classification report — not just the bottom-line number — before declaring a model ready.

## What you learned

- Neural networks are stacked layers of weighted connections with non-linear activations between them
- Keras lets you build a network by assembling layers, and `model.summary()` shows the architecture and parameter count
- `compile` sets the optimizer, loss, and metrics; `fit` runs the training loop and returns a history you can plot
- Plotting train versus validation curves reveals overfitting as the two lines diverge
- `EarlyStopping` stops training when validation performance peaks — the neural network equivalent of `max_depth`
- Confusion matrices on multi-class problems reveal which specific errors the model makes, not just how many

## Stretch goals

- Add `layers.Dropout(0.3)` after each hidden layer and compare train vs validation accuracy curves to the version without dropout. What does the gap look like?
- Try the `RMSprop` optimizer instead of `adam` and compare convergence speed on the loss curve.
- Visualize what the first hidden layer learned: extract `best_model.layers[0].get_weights()[0]`, reshape each of the 64 columns into an 8×8 image, and plot them — each image is a pattern detector the network discovered.
