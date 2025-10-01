import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Hide TensorFlow warnings

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
# MNIST dataset contains 28x28 grayscale images of digits (0-9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1] for better training
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to include channel dimension (required for Keras)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build a simple Convolutional Neural Network (CNN) model
model = models.Sequential([
    # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling to reduce spatial dimensions
    layers.MaxPooling2D((2, 2)),
    # Another convolutional layer with 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Flatten the output for dense layers
    layers.Flatten(),
    # Dense layer with 128 neurons
    layers.Dense(128, activation='relu'),
    # Output layer with 10 neurons (one for each digit) and softmax activation
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary to understand its structure
model.summary()

# Train the model for 5 epochs with a batch size of 32
history = model.fit(x_train, y_train, epochs=5, batch_size=32, 
                    validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")

# Visualize some predictions (optional, for understanding)
# Predict on a few test images
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test[:5], axis=1)

# Display the first 5 test images with predicted and actual labels
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {actual_labels[i]}")
    plt.axis('off')
plt.show()