import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam


###########  TINY VGG ARCHITECTURE  ###################

# set random seed
tf.random.set_seed(42)

# Create the model
model_3 = Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=1,
            kernel_initializer="he_uniform",
            activation="relu",
        ),
        layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=1,
            kernel_initializer="he_uniform",
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=1,
            kernel_initializer="he_uniform",
            activation="relu",
        ),
        layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=1,
            kernel_initializer="he_uniform",
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="model_3",
)

# Compile the model
model_3.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(), metrics=["accuracy"]
)

########### FIT THE MODEL FOR 5 EPOCHS  #################

###########  ON NORMAL DATA  ####################
# Fit the model
model_3_history = model_3.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=len(train_dataset),
    validation_data=test_dataset,
    validation_steps=len(test_dataset),
)

# Evaluate the model on test data
model_3.evaluate(test_dataset)

# Plot the loss curves
plot_loss_curves(model_3_history)


############  ON AUGMENTED DATA  #######################
# Fit the model
model_4_b_history = model_4_b.fit(
    train_aug_dataset,
    epochs=5,
    steps_per_epoch=len(train_aug_dataset),
    validation_data=test_dataset,
    validation_steps=len(test_dataset),
)

model_4_b.evaluate(test_dataset)

# Plot loss curves
plot_loss_curves(model_4_b_history)
