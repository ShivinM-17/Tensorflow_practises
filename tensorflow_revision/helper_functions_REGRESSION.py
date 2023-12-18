import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


# Making a function to get model results
def get_model_results(model, test_data):
    X_test, y_test = test_data

    # Predict using the data
    model_preds = tf.squeeze(model.predict(X_test), axis=-1)

    # Get evaluation metrics
    mae = tf.keras.metrics.mae(y_test, model_preds).numpy()
    mse = tf.keras.metrics.mse(y_test, model_preds).numpy()
    rmse = tf.math.sqrt(mse).numpy()

    return {"mae": mae, "mse": mse, "rmse": rmse}, (model_preds)


# Plot prediction Vs Actual value
def plot_pred_vs_act(y_true, y_pred, start=0):
    plt.figure(figsize=(10, 7))
    plt.plot(tf.range(len(y_true))[start:], y_true[start:], label="Testing values")
    plt.plot(tf.range(len(y_true))[start:], y_pred[start:], label="Prediction values")
    plt.xlabel("Records")
    plt.ylabel("Prices")
    plt.legend()
    plt.grid(True)


# Plot the loss function
def plot_loss_curves(history, metrics_to_plot=["loss"]):
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 7))
        plt.plot(pd.DataFrame(history.history[metric]), label=f"Training {metric}")
        plt.plot(
            pd.DataFrame(history.history["val_" + metric]), label=f"Validation {metric}"
        )
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()


# Making a function to fit the model at different epochs and get the best weight model metrics
# incorporate LR rate scheduler, earlystopping and modelcheckpoint

def create_model_checkpoint(save_path="model_exp"):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path, monitor="val_loss", save_best_only=True, verbose=0
    )


def create_lr_scheduler(epoch=80, schedule="", custom=False):
    gen_schedule = lambda epochs: 1e-5 * 10 ** (epochs / int(epoch * 0.25))
    return tf.keras.callbacks.LearningRateScheduler(
        schedule if custom else gen_schedule, verbose=0
    )


def create_early_stopping(patience=32):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )


def get_best_model(
    model,
    train_data,
    test_data,
    compile_loss="mae",
    epoch_list: list = [64, 128, 256, 512],
    checkpt=True,
    lr_schedule=False,
    lr_cust_schedule=None,
    earlystop=False,
):
    # set random seed
    tf.random.set_seed(42)

    model_list = {}
    model_results = {}
    model_historys = {}
    model_preds = {}

    for epoch in epoch_list:
        MODEL_PATH = f"model_exp/{model.name}/{model.name}_{epoch}"
        model_ep = tf.keras.models.clone_model(model)

        # Get the model checkpoint
        callbacks = [create_model_checkpoint(save_path=MODEL_PATH)] if checkpt else []

        # Get Early stopping callback
        if earlystop:
            callbacks = [create_early_stopping(patience=epoch * 0.20)]

        # Get the learning rate scheduler
        if lr_schedule and epoch > 200:
            callbacks += (
                [
                    create_lr_scheduler(
                        schedule=lr_cust_schedule, epoch=epoch, custom=True
                    )
                ]
                if lr_cust_schedule
                else [create_lr_scheduler(epoch)]
            )

        # Compile the model
        model_ep.compile(loss=compile_loss, optimizer=Adam(), metrics=["mae", "mse"])

        # Fit the model
        model_ep_history = model_ep.fit(
            train_data[0],
            train_data[1],
            epochs=epoch,
            validation_data=test_data,
            verbose=0,
            callbacks=callbacks,
        )

        model_historys[f"{model.name}_{epoch}"] = model_ep_history
        print(f"Model has been trained on {epoch} epochs.")

        # Check if ModelCheckpoint callback done or not
        if checkpt:
            model_ep = tf.keras.models.load_model(MODEL_PATH)

        # Get evaluation results of the model
        print(f"Evaluating the model on test data")
        model_ep_results, model_ep_preds = get_model_results(model_ep, test_data)

        model_results[f"{model.name}_{epoch}"] = model_ep_results
        model_preds[f"{model.name}_{epoch}"] = model_ep_preds
        model_list[f"{model.name}_{epoch}"] = model_ep
        print()

    model_results = pd.DataFrame(model_results).T.sort_values(
        ["mae", "mse", "rmse"], ascending=[True, True, True]
    )

    return model_list, model_results, model_historys, model_preds


# Make function to plot learning rate Vs Loss
def plot_lr_loss(history, lrs):
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Loss Vs Learning rate")
    plt.axis(True)
