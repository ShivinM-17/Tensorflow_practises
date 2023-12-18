############ UNZIP DATA FUNCTION  ################
# Making a function to unzip a zipfile
import zipfile


def unzip_data(zip_dir):
    zip_ref = zipfile.ZipFile(zip_dir)
    zip_ref.extractall()
    zip_ref.close()


# eg.
# unzip the data file
unzip_data("kagglecatsanddogs_5340.zip")


#############   PLOT LOSS CURVES    #######################
# Plot loss curves of the model
import matplotlib.pyplot as plt


def plot_loss_curves(history, metrics_to_plot=["loss", "accuracy"]):
    plt.figure(figsize=(10, 10))
    for index, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(2, 1, index + 1)
        plt.plot(history.history[metric], label=f"training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        plt.title(f"{metric} vs Epochs curve")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()


############# COMPARE HISTORIES AND PLOT #######################
def compare_history(
    original_history,
    new_history,
    initial_epochs=5,
    metrics_to_plot=["loss", "accuracy"],
):
    plt.figure(figsize=(7, 7))
    for index, metric in enumerate(metrics_to_plot):
        mtr = original_history.history[metric]
        val_mtr = original_history.history[f"val_{metric}"]

        total_mtr = mtr + new_history.history[metric]
        total_val_mtr = val_mtr + new_history.history[f"val_{metric}"]

        plt.subplot(2, 1, index + 1)
        plt.plot(total_mtr, label=f"Training {metric}")
        plt.plot(total_val_mtr, label=f"Validation {metric}")
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim())
        plt.legend()
        plt.title(f"Training and validation {metric}")


############# COMPARE ALL HISTORIES (MANY) AND PLOT #######################
# Making a function to compare histories at any time
def compare_history_overall(
    history_lst=[], initial_epoch_lst=[5], metrics_to_plot=["loss", "accuracy"]
):
    plt.figure(figsize=(12, 12))
    for index, metric in enumerate(metrics_to_plot):
        total_mtr = history_lst[0].history[metric]
        total_val_mtr = history_lst[0].history[f"val_{metric}"]

        for history in history_lst[1:]:
            total_mtr = total_mtr + history.history[metric]
            total_val_mtr = total_val_mtr + history.history[f"val_{metric}"]

        plt.subplot(2, 1, index + 1)
        plt.plot(total_mtr, label=f"Training {metric}")
        plt.plot(total_val_mtr, label=f"Validation {metric}")

        for epoch in initial_epoch_lst:
            plt.plot([epoch - 1, epoch - 1], plt.ylim())

        plt.title(f"Training and validation {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.legend()


######## GET MODEL RESULTS AT DIFFERENT EPOCHS WITH HISTORIES AND EVALUATION METRICS #####

### MAKING A FUNCTION THAT WILL FIT A MODEL TILL SPECIFIC EPOCH POINTS,
### THEN CONTINUE FROM THAT EPOCH WITH THE BEST WEIGHTS MODEL
import tensorflow as tf


def get_range_epoch_results(
    model,
    epoch_lst=[5, 20],
    callbacks=None,
    loss_fn="binary_crossentropy",
    train_data=train_data,
    test_data=test_data,
):
    history_lst = []
    eval_res = []

    epoch_lst = [0] + epoch_lst
    for index, epoch in enumerate(epoch_lst[1:]):
        if index == 0:
            model.compile(loss=loss_fn, optimizer=Adam(), metrics=["accuracy"])

        # Make the callback list
        if callbacks:
            callbacks += [
                create_model_checkpoint(save_dir=f"model_exp/model_rng_{index}")
            ]
        else:
            callbacks = [
                create_model_checkpoint(save_dir=f"model_exp/model_rng_{index}")
            ]

        # Set the initial_epoch and train the data from it
        initial_epoch = epoch_lst[index]
        print(
            f"Fitting and training the model: \n  Initial epochs: {initial_epoch}\n  Total epochs: {epoch}\n"
        )
        model_history = model.fit(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=epoch,
            initial_epoch=initial_epoch,
            validation_data=test_data,
            validation_steps=len(test_data),
            verbose=0,
            callbacks=callbacks,
        )

        print(f"Trained the model till {epoch} epochs!!!\n")

        # Load in the best model
        print(f"Loading the best weights model {index + 1}")
        model = tf.keras.models.load_model(f"model_exp/model_rng_{index}")

        # Evaluating the model
        print("Getting the evaluation results")
        eval_res.append(model.evaluate(test_data))

        history_lst.append(model_history)

        print()
        print("-" * 50)
        print()

    return history_lst, eval_res
