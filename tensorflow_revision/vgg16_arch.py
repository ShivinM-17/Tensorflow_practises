import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam

######### VGG16 ARCHITECTURE   #####################
# set random seed
tf.random.set_seed(42)

# Create the model
model_2 = Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        # Layer 1
        layers.Conv2D(
            filters=8, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.Conv2D(
            filters=8, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # Layer 2
        layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # Layer 3
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # Layer 4
        layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", strides=1, activation="relu"
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # Layer 5
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=1,
            activation="relu",
        ),
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=1,
            activation="relu",
        ),
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=1,
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # Layer 6
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=1,
            activation="relu",
        ),
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=1,
            activation="relu",
        ),
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=1,
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # Layer 7
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="model_2",
)

# Compile the model
model_2.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"]
)

model_2.summary()

################ TRAINING FOR 5 EPOCHS  #####################

# Fit the model
model_2_history = model_2.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=len(train_dataset),
    validation_data=test_dataset,
    validation_steps=len(test_dataset),
)

# Evaluate on test data
model_2.evaluate(test_dataset)

# Plot loss curves
plot_loss_curves(model_2_history)


############ TRAINING FOR 10 MORE EPOCHS  ###################

# Fit for 10 more epochs
model_2_history_new = model_2.fit(
    train_dataset,
    epochs=15,
    initial_epoch=5,
    steps_per_epoch=len(train_dataset),
    validation_data=test_dataset,
    validation_steps=len(test_dataset),
)


# Evaluate on test data
model_2.evaluate(test_dataset)

compare_historys(model_2_history, model_2_history_new)
