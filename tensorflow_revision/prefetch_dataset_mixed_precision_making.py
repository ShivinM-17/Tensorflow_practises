########### MAKING PREFETCHED  DATASETS  ##############

## HERE, USING CATS VS DOGS ##

# Getting catsVsDogs dataset using tfds
import tensorflow_datasets as tfds
import tensorflow as tf

(train, test), metadata = tfds.load(
    "cats_vs_dogs",
    split=["train[:85%]", "train[85%:]"],
    with_info=True,
    shuffle_files=True,
)

#### GETTING MORE INFORMATION OUT OF THE DATA  ####

# Get features dictionary of the data
metadata.features

# Get number of classes and class names
NUM_CLASSES = metadata.features["label"].num_classes
class_names = metadata.features["label"].names
NUM_CLASSES, class_names

# Checking the shape of the data
metadata.features.shape, metadata.features.dtype

# Count of samples in train and test data
print("Number of samples in train data:", metadata.splits["train[:85%]"].num_examples)
print("Number of samples in test data:", metadata.splits["train[85%:]"].num_examples)


###### MAKING A PREPROCESSING FUNCTION FOR THE IMAGES #####
# Make a function for preprocessing images
def preprocess_img(sample, img_shape=224):
    image = sample["image"]
    image = tf.image.resize(image, [img_shape, img_shape])  # reshape to img_shape
    return (tf.cast(image, tf.float32), sample["label"])


##### BATCHING AND PREPARING THE DATASET  ########
# Make the train data
train_data = train.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = (
    train_data.shuffle(buffer_size=1000)
    .batch(batch_size=16)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Make the train data
test_data = test.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE)


###### SET UP MIXED PRECISION TRAINING  ######
# Turn on mixed precision training
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(
    policy="mixed_float16"
)  # set global policy to mixed precision
