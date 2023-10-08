# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:57:59 2023

@author: Iqra Asghar

Deep Learning Assignment-01 Classification using KNN

"""

import cv2
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(train_data, train_labels, test_data, k=1):
    predictions = []

    for test_sample in test_data:
        distances = [euclidean_distance(test_sample, train_sample) for train_sample in train_data]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [train_labels[i] for i in nearest_indices]
        prediction = np.argmax(np.bincount(nearest_labels))
        predictions.append(prediction)

    return predictions


with_candy_images = []
for i in range(10):
    img = cv2.imread(f'with_candy_images/{i}.jpg')
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    with_candy_images.append((img, 1))

without_candy_images = []
for i in range(10):
    img = cv2.imread(f'without_candy_images/{i}.jpg')
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    without_candy_images.append((img, 0))  # Change label to 0 for "without candy"

# Combining and shuffling the datasets
all_images = with_candy_images + without_candy_images
np.random.shuffle(all_images)

# Separating data and labels
data = np.array([item[0] for item in all_images])
labels = np.array([item[1] for item in all_images])

# Splitting the data into training 70% and testing 30%
split_ratio = 0.7
split_index = int(split_ratio * len(data))

train_data = data[:split_index]
train_labels = labels[:split_index]
test_data = data[split_index:]
test_labels = labels[split_index:]

# Using k-NN to predict labels for the testing data
k = 1
predictions = knn_predict(train_data, train_labels, test_data, k)

# Calculating accuracy
correct_predictions = np.sum(predictions == test_labels)
total_predictions = len(test_labels)
accuracy = correct_predictions / total_predictions * 100

print(f"Accuracy: {accuracy:.2f}%")

test_img = cv2.imread('test_image.jpg')
test_img = cv2.resize(test_img, (64, 64))
test_data = test_img.flatten()

# Using k-NN to predict if the image has candy or not
prediction = knn_predict(train_data, train_labels, [test_data], k)[0]

if prediction == 1:
    print("The image has candy.")
else:
    print("The image does not have candy.")
