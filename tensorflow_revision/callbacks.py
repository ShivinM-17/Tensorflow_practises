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


# Stop training at particular accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("acc") > 0.90:
            print("\nReached 90% accuracy - Cancelling training -")
            self.model.stop_training = True
