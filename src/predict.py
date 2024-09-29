import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils  # Importing utility functions

# Load the model
model = tf.keras.models.load_model('models/cnn_cifar10.h5')

# Load CIFAR-10 dataset (for testing predictions)
(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Preprocess the test images
test_images = utils.preprocess_data(test_images)

# Make predictions on test data
predictions = model.predict(test_images)

# Display some sample predictions
num_images = 5
for i in range(num_images):
    utils.plot_image(i, predictions, test_labels, test_images)
