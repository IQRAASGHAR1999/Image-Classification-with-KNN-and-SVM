# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:57:59 2023

@author: Iqra Asghar

Deep Learning Assignment-01 Classification using SVM

"""

import cv2
import numpy as np

def svm_train(X, y, learning_rate=0.01, num_epochs=1000):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for epoch in range(num_epochs):
        for i in range(num_samples):
            condition = y[i] * (np.dot(X[i], weights) + bias)
            
            if epoch == 0:
                epoch_divisor = 1  # Avoid division by zero in the first iteration
            else:
                epoch_divisor = epoch
            
            if condition >= 1:
                weights -= learning_rate * (2 / epoch_divisor * weights)
            else:
                weights -= learning_rate * (2 / epoch_divisor * weights - np.dot(X[i], y[i]))
                bias -= learning_rate * y[i]

    return weights, bias

def svm_predict(X, weights, bias):
    prediction = np.dot(X, weights) + bias
    return np.sign(prediction)
#importing dataset
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
    without_candy_images.append((img, -1))  

# Combining and shuffling the datasets
all_images = with_candy_images + without_candy_images
np.random.shuffle(all_images)

# Separating data and labels
data = np.array([item[0] for item in all_images])
labels = np.array([item[1] for item in all_images])

# Spliting the data into training 70% and testing 30%
split_ratio = 0.6
split_index = int(split_ratio * len(data))

train_data = data[:split_index]
train_labels = labels[:split_index]
test_data = data[split_index:]
test_labels = labels[split_index:]

# Training the SVM model on the training data
learning_rate = 0.01
num_epochs = 1000
weights, bias = svm_train(train_data, train_labels, learning_rate, num_epochs)


predictions = svm_predict(test_data, weights, bias)

# Calculating accuracy
correct_predictions = np.sum(predictions == test_labels)
total_predictions = len(test_labels)
accuracy = correct_predictions / total_predictions * 100

print(f"Accuracy: {accuracy:.2f}%")
test_img = cv2.imread('test_image.jpg')
test_img = cv2.resize(test_img, (64, 64))
test_data = test_img.flatten()

# Predicting if the image has candy or not
prediction = svm_predict(test_data, weights, bias)

if prediction == 1:
    print("The image has candy.")
else:
    print("The image does not have candy.")