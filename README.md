# CNN Image Classification

This project uses a Convolutional Neural Network (CNN) to classify images using the CIFAR-10 dataset. The dataset contains 60,000 32x32 color images in 10 different classes. The model is trained using the TensorFlow and Keras libraries.

## Project Structure
```python

CNN-Image-Classification/
│
├── data/                   # Dataset (CIFAR-10 and MNIST loaded using keras)
│
├── src/                    # Source code for training and evaluation
│   ├── train.py            # Code to train the CNN
│   ├── predict.py          # Code for prediction using the trained model
│   ├── utils.py            # Utility functions for data preprocessing, etc.
│
├── models/                 # Save trained models
│   ├── cnn_mnist.h5        # Trained CNN model for MNIST
│   ├── cnn_cifar10.h5      # Trained CNN model for CIFAR-10
│
├── notebooks/              # Jupyter notebooks for interactive exploration
│   ├── cnn_mnist.ipynb     # Notebook for MNIST dataset
│   ├── cnn_cifar10.ipynb   # Notebook for CIFAR-10 dataset
│
├── plots/                  # Save training plots (accuracy, loss, confusion matrix)
│
├── requirements.txt        # List of dependencies (tensorflow, keras, numpy, etc.)
├── README.md               # Project description and setup instructions

```
# Project description and setup instructions

## Libraries Used
- **TensorFlow**: For building the CNN model.
- **Keras**: High-level API for deep learning.
- **NumPy**: For data manipulation.
- **Matplotlib**: For visualizing accuracy and loss curves.

## Instructions

1. Install the required dependencies:

  ```python
    pip install -r requirements.txt
  ```

2. Train the model:

  ```python
   python src/train.py
  ```
3. Test the model and make predictions: 

```python
 python src/predict.py
```

4. You can explore the dataset and model in the `notebooks/cnn_cifar10.ipynb` Jupyter notebook.

### Dummy Data (for CIFAR-10)
The CIFAR-10 dataset is loaded using Keras’ datasets API, so there is no need to manually download it
