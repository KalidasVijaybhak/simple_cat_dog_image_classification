# Cat and Dog Image Classifier using Convolutional Neural Network (CNN)
https://classify-cat-dog.streamlit.app/ 

This project implements a Convolutional Neural Network (CNN) model to classify images as either cats or dogs. The model is trained on a dataset containing labeled images of cats and dogs, and achieves competitive performance in distinguishing between the two classes.

## Overview

This CNN model is built using [insert framework/library here, e.g., TensorFlow, PyTorch] and trained on a dataset comprising thousands of cat and dog images. The goal of the model is to accurately predict whether a given image contains a cat or a dog.

## Model Architecture

The CNN architecture used for this project consists of several convolutional layers followed by max-pooling layers to extract and learn features from the input images. The final layers are fully connected to make predictions based on the learned features.

- **Input Layer:** Accepts input images of size [specify dimensions].
- **Convolutional Layers:** [Brief description of layers and filters used].
- **Pooling Layers:** Max-pooling layers to downsample feature maps.
- **Fully Connected Layers:** Dense layers for classification.

## Dataset

The model is trained on a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/sunilthite/cat-or-dog-image-classification/data). It includes a balanced collection of cat and dog images, each labeled accordingly.


## Model Definition

- **Activation Function:** ReLU is used in the convolutional layers.
- **Output Activation Function:** Sigmoid is used in the final dense layer for binary classification.

## Training Configuration

- **Optimizer:** Adam optimizer is used (`'adam'`).
- **Loss Function:** Binary Cross-Entropy (`'binary_crossentropy'`) is used.
- **Metrics:** Accuracy is used to evaluate the model's performance.
- **Training Steps:** 11 epochs with a batch size of 64.


## Future Improvements

Potential enhancements and future work include:
- Experimenting with different architectures (e.g., transfer learning).
- Enhancing dataset diversity.
- Improving model robustness and efficiency.

## Sample Predictions

![image](https://github.com/KalidasVijaybhak/simple_cat_dog_image_classification/assets/70281178/071c7c67-dc00-4567-a9cf-ab098d820096)
*Screenshot*

![image](https://github.com/KalidasVijaybhak/simple_cat_dog_image_classification/assets/70281178/67f47c2c-60c9-489b-a53e-1c810fed372d)
*Model*
