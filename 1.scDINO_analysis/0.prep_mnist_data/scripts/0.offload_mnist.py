#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pickle
import sys

import mnist
import numpy as np

sys.path.append("../../../utils")
import matplotlib.pyplot as plt
import tifffile
import torch
import tqdm
from prep_data import MNIST_Dataset
from torch.utils.data import DataLoader

# In[2]:


# open the data
# set path to the data
mnist_pickle_path = pathlib.Path("../../../data/mnist/MNIST.pkl").resolve(strict=True)
photos_output_path_train = pathlib.Path("../../../data/mnist_photos/train").resolve()
photos_output_path_test = pathlib.Path("../../../data/mnist_photos/test").resolve()
photos_output_path_train.mkdir(parents=True, exist_ok=True)
photos_output_path_test.mkdir(parents=True, exist_ok=True)

with open(mnist_pickle_path, "rb") as file_handle:
    MNIST = pickle.load(file_handle)

# set the batch size
batch_size = 100
subset_size = None
# number of frames to generate
num_frames = 25

# In[3]:


def del_files(folder):
    # get the number of files in the folder
    while len(list(folder.iterdir())) > 0:
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # check for files in the folder
                if len(list(file.iterdir())) == 0:
                    file.rmdir()
                elif len(list(file.iterdir())) > 0:
                    del_files(file)
            else:
                del_files(file)


del_files(photos_output_path_train)
del_files(photos_output_path_test)

# ### Get the train data

# In[4]:


# create the data class
# this class makes a rolling window of the data
data = MNIST_Dataset(
    MNIST["train_image"],
    MNIST["train_label"],
    binary=False,
    number_of_frames=num_frames,
    subset=subset_size,
)
data.label.shape

# In[5]:


# create the data loader
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
# get the first batch of data
batch, sample = next(enumerate(data_loader))
print(f"There will be {data.label.shape[0] / batch_size} total batches.")
print(f"Each batch will have {batch_size} images.")
print("However, this is temporal data, so each image will have multiple frames.")
print(f"Each image will have {sample[1]['image'].shape[1]} frames.")
print(f"Each frame will have {sample[1]['image'].shape[2]} pixels.")
print(
    f"For a total of {data.label.shape[0] * sample[1]['image'].shape[1]} final images."
)
count = 0
# loop through the data and save the images to disk
for i in tqdm.tqdm(range(int(data.label.shape[0] / batch_size))):
    count = count + (1 * batch_size)
    # get the batch
    batch, sample = next(enumerate(data_loader))
    # loop through the batch
    for j in range(batch_size):
        label = sample[1]["label"].tolist()[1][0][0]
        sample_image = sample[1]["image"][j].numpy()
        idx = sample[0].tolist()[j]
        for k in range(sample[1]["image"][j].shape[0]):
            # save the image
            photos_output_file_path_train = pathlib.Path(
                photos_output_path_train / f"label_{i}_idx_{idx}_timepoint_{k}"
            )
            photos_output_file_path_train.mkdir(parents=True, exist_ok=True)
            # define a blank image
            blank_image = np.zeros((28, 28))
            # save both images as a multi channel image tiff
            multi_channel_image = np.stack(
                [sample[1]["image"][j][k].reshape(28, 28), blank_image], axis=0
            )
            # ensure that the image is saved as a tiff in the following format:
            # (channel, x, y)
            multi_channel_image = np.moveaxis(multi_channel_image, 0, -1)
            tifffile.imsave(
                photos_output_file_path_train
                / f"label_{label}_idx_{idx}_timepoint_{k}.tiff",
                multi_channel_image,
            )


print(count)

# ### Get the test data

# In[6]:


# create the data class
# this class makes a rolling window of the data
data = MNIST_Dataset(
    MNIST["test_image"],
    MNIST["test_label"],
    binary=False,
    number_of_frames=num_frames,
    subset=100,
)
# create the data loader
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
# get the first batch of data
batch, sample = next(enumerate(data_loader))
print(f"There will be {data.label.shape[0] / batch_size} total batches.")
print(f"Each batch will have {batch_size} images.")
print("However, this is temporal data, so each image will have multiple frames.")
print(f"Each image will have {sample[1]['image'].shape[1]} frames.")
print(f"Each frame will have {sample[1]['image'].shape[2]} pixels.")
print(
    f"For a total of {data.label.shape[0] * sample[1]['image'].shape[1]} final images."
)
count = 0
# loop through the data and save the images to disk
for i in tqdm.tqdm(range(int(data.label.shape[0] / batch_size))):
    count = count + (1 * batch_size)
    # get the batch
    batch, sample = next(enumerate(data_loader))
    # loop through the batch
    for j in range(batch_size):
        # print(sample[0].tolist()[j])
        sample_image = sample[1]["image"][j].numpy()
        idx = sample[0].tolist()[j]
        for k in range(sample[1]["image"][j].shape[0]):
            # save the image
            photos_output_file_path_test = pathlib.Path(
                photos_output_path_test / f"label_{i}_idx_{idx}_timepoint_{k}"
            )
            photos_output_file_path_test.mkdir(parents=True, exist_ok=True)
            # define a blank image
            blank_image = np.zeros((28, 28))
            # save both images as a multi channel image tiff
            multi_channel_image = np.stack(
                [sample[1]["image"][j][k].reshape(28, 28), blank_image], axis=0
            )
            # ensure that the image is saved as a tiff in the following format:
            # (channel, x, y)
            multi_channel_image = np.moveaxis(multi_channel_image, 0, -1)
            tifffile.imsave(
                photos_output_file_path_test
                / f"label_{label}_idx_{idx}_timepoint_{k}.tiff",
                multi_channel_image,
            )

print(count)
