import matplotlib.pyplot as plt


# Make a function to plot loss curves
def plot_loss_curves(history, metrics_to_plot=["loss", "accuracy"]):
    plt.figure(figsize=(9, 9))
    for index, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(2, 1, index + 1)
        plt.plot(history.history[metric], label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(f"{metric} Vs Epochs curve")
        plt.subplots_adjust(hspace=0.6)
        plt.grid(True)
        plt.legend()


# Get random images
import random


def plot_random_image(train_data, train_labels, class_names):
    plt.figure(figsize=(7, 7))
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        rand_index = random.choice(range(len(train_data)))
        plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
        plt.title(class_names[train_labels[rand_index]])
        plt.axis(False)


# Plotting and checking the predictions made by the model
import random


def plot_pred_images(y_preds, test_data, test_labels, class_names):
    plt.figure(figsize=(10, 10))
    col = "r"
    for index in range(4):
        ax = plt.subplot(2, 2, index + 1)
        rand_index = random.choice(range(len(y_preds)))
        pred_class = class_names[y_preds[rand_index]]
        actual_class = class_names[test_labels[rand_index]]

        plt.imshow(test_data[rand_index], cmap=plt.cm.binary)

        if pred_class == actual_class:
            col = "g"
        else:
            col = "r"

        plt.title(f"Predicted: {pred_class}; Actual: {actual_class}", color=col)
        plt.axis(False)
