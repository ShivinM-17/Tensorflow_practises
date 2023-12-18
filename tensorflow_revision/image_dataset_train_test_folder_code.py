######## MAKING TRAIN AND TEST DIRECTORIES FOR IMAGE DATA  ########
import os
from shutil import copyfile
import random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image


# Making a function to check if file is corrupt or not
def is_file_corrupt(filename):
    try:
        with Image.open(filename) as img:
            img.verify()
        return False
    except Exception as e:
        return True


# Making a function to split the data into train and test splits
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        try:
            if (
                os.path.getsize(file) > 0
                and not is_file_corrupt(file)
                and filename[-3:] == "jpg"
            ):
                files.append(filename)
            else:
                print(filename + " is zero length / corrupt, so ignoring.")
        except Exception as e:
            print("Corrupt file, ignoring it")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[-1:-training_length:-1]
    testing_set = shuffled_set[:testing_length]

    if not os.path.exists(TRAINING):
        os.makedirs(TRAINING)

    if not os.path.exists(TESTING):
        os.makedirs(TESTING)

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


######### FOR CATS VS DOGS DATASET  ###################
# Get the train and test directories to store the data in
CAT_SOURCE_DIR = "/content/PetImages/Cat/"
TRAINING_CATS_DIR = "/content/pet_data/train/cats/"
TESTING_CATS_DIR = "/content/pet_data/test/cats/"
DOG_SOURCE_DIR = "/content/PetImages/Dog/"
TRAINING_DOGS_DIR = "/content/pet_data/train/dogs/"
TESTING_DOGS_DIR = "/content/pet_data/test/dogs/"

# Now, make the file and folder splits
split_size = 0.85
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Check the folder structure
# Walk through pet_data directory and list number of files
for dirpath, dirnames, filenames in os.walk("pet_data"):
    print(
        f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
    )


# Get class names
import pathlib
import numpy as np

data_dir = pathlib.Path("pet_data/train/")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)
