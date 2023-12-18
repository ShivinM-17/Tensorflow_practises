import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


############## MAKING THE IMAGE_DATA_GENERATORS ##################
# Set the random seed
tf.random.set_seed(42)

# Making the train images generator
train_datagen = ImageDataGenerator(rescale=1 / 255.0)

# Making the augmented train images generator
train_datagen_aug = ImageDataGenerator(
    rescale=1 / 255.0,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    rotation_range=15,
)

# Making the test image generator
test_datagen = ImageDataGenerator(rescale=1 / 255.0)


############  USING THE GENERATORS TO MAKE THE DATASETS ####################
train_dir = "/content/pet_data/train/"
test_dir = "/content/pet_data/test/"

# Making the train dataset
train_dataset = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=16, class_mode="binary", seed=42
)
# Making the train augmented dataset
train_aug_dataset = train_datagen_aug.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=16, class_mode="binary", seed=42
)
# Making the test dataset
test_dataset = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=16, class_mode="binary", seed=42
)
