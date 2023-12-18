from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam


########## Creating the neural network  #################
# Set the random seed
tf.random.set_seed(42)

# Create model
model_1 = Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(15, 3, padding="valid", activation="relu"),
        layers.MaxPool2D(pool_size=2, padding="valid"),
        layers.Conv2D(10, 3, padding="valid", activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(10, 3, padding="valid", activation="relu"),
        layers.Conv2D(10, 3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="model_1",
)

# Compile the model
model_1.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"]
)

# Fit the model
model_1_history = model_1.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=len(train_dataset),
    validation_data=test_dataset,
    validation_steps=len(test_dataset),
)
