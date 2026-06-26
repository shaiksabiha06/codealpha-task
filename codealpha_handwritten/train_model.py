import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


print("=" * 60)
print("Loading EMNIST Balanced Dataset...")
print("=" * 60)


train = pd.read_csv(
    "dataset/emnist-balanced-train.csv",
    header=None
)

test = pd.read_csv(
    "dataset/emnist-balanced-test.csv",
    header=None
)


print("Dataset Loaded Successfully!")


X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values


print("Preprocessing Images...")


X_train = X_train.reshape(-1,28,28,1).astype("float32") / 255.0
X_test = X_test.reshape(-1,28,28,1).astype("float32") / 255.0


num_classes = len(np.unique(y_train))


y_train = to_categorical(
    y_train,
    num_classes
)

y_test = to_categorical(
    y_test,
    num_classes
)


print("Building CNN Model...")


model = Sequential()


model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)


model.add(
    MaxPooling2D((2,2))
)


model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)


model.add(
    MaxPooling2D((2,2))
)


model.add(
    Flatten()
)


model.add(
    Dense(
        256,
        activation="relu"
    )
)


model.add(
    Dropout(0.5)
)


model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)



# NEW FORMAT
checkpoint = ModelCheckpoint(
    "model.keras",
    save_best_only=True
)



print("\nTraining Started...\n")


history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=[early_stop, checkpoint]
)



print("\nEvaluating Model...")


loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print("=" * 60)
print(f"Test Accuracy : {accuracy*100:.2f}%")
print("=" * 60)



plt.figure(figsize=(10,4))


plt.subplot(1,2,1)

plt.plot(
    history.history["accuracy"]
)

plt.plot(
    history.history["val_accuracy"]
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(
    ["Train","Validation"]
)



plt.subplot(1,2,2)

plt.plot(
    history.history["loss"]
)

plt.plot(
    history.history["val_loss"]
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(
    ["Train","Validation"]
)


plt.tight_layout()

plt.show()



print("\nModel Saved Successfully as model.keras")