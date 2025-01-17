# Skin Lesions Detection with CNN
## Overview

The Skin Lesion Detection project leverages machine learning techniques to identify and classify skin lesions, such as melanoma, from medical images. The aim is to aid early diagnosis and improve patient outcomes by providing a reliable and efficient automated tool for dermatological analysis.

**Features**

Automated classification of skin lesions into multiple categories (e.g., benign, malignant).

Preprocessing techniques for image enhancement and noise reduction.

Implementation of Convolutional Neural Networks (CNNs) for image analysis.

Evaluation metrics, including accuracy, precision, recall, and F1-score, to assess model performance.

User-friendly interface for uploading images and viewing results.

**Technologies Used**

Programming Languages: Python

Frameworks/Libraries: TensorFlow, Keras, OpenCV, Scikit-learn

Tools: Jupyter Notebook, Matplotlib, NumPy, Pandas

Dataset: Publicly available skin lesion datasets such as ISIC (International Skin Imaging Collaboration)

### **Dataset**

---
The dataset used for training and testing is sourced from the ISIC Archive. It contains labeled dermoscopic images of various skin lesions. Ensure the dataset is downloaded and structured as follows:

/dataset
  /train
    /class_1
    /class_2
    ...
  /test
    /class_1
    /class_2
    ...

Installation

Clone the repository:

git clone https://github.com/username/skin-lesion-detection.git

Navigate to the project directory:- cd skin-lesion-detection

Install dependencies:- pip install -r requirements.txt


---


## Usage
Preprocess the dataset: Run the preprocessing script to resize and normalize images.

python preprocess.py

Train the model:Train the CNN model using the prepared dataset.

python train.py

Evaluate the model:Test the trained model and view the performance metrics.

python evaluate.py

Detect lesions:Use the trained model to classify new images.

## Future Work
Expand the model to include more classes of skin conditions.

Improve accuracy using advanced architectures like EfficientNet or Vision Transformers.

Deploy the model as a web application for real-time diagnosis.

**Contributing**

Contributions are welcome! Feel free to fork the repository and submit pull requests.

**License**

This project is licensed under the MIT License.

## Acknowledgements

The ISIC Archive for providing the dataset.

TensorFlow and Keras documentation for guidance on model implementation.


```python
for layer in pre_trained_model.layers:
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False
```

## Results

| Models        | Validation           | Test            |  Depth          | # Params          |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:|
|   Baseline   | 77.48% |76.54% | 11 layers | 2,124,839 |
|  Fine-tuned VGG16 (from last block)    |  79.82%      |   79.64%  | 23 layers | 14,980,935 |
|  Fine-tuned Inception V3 (from the last 2 inception blocks) |  79.935%   |  79.94% | 315 layers | 22,855,463 |
|  Fine-tuned Inception-ResNet V2 (from the Inception-ResNet-C) | 80.82% | 82.53% | 784 layers | 55,127,271 |
|  Fine-tuned DenseNet 201 (from the last dense block) | **85.8%** | **83.9%**  |  711 layers | 19,309,127 |
|  Fine-tuned Inception V3 (all layers) | 86.92% | 86.826% | _ | _ |
|  Fine-tuned DenseNet 201 (all layers)  | **86.696%** | **87.725%** | _ | _ |
|  Ensemble of fully-fine-tuned Inception V3 and DenseNet 201 | **88.8%** | **88.52%** | _ | _ |




