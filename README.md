# 🔢 MNIST Digit Recognition using CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A deep learning project that recognizes handwritten digits (0-9) using Convolutional Neural Networks (CNN) trained on the MNIST dataset. This implementation achieves over 98% accuracy on test data.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the famous MNIST dataset. The model is built using TensorFlow/Keras and demonstrates fundamental concepts of deep learning and computer vision.

### Why This Project?
- 🎓 Perfect for learning deep learning fundamentals
- 🚀 Industry-standard implementation
- 📊 High accuracy with minimal complexity
- 💡 Easy to understand and modify

## ✨ Features

- ✅ **CNN Architecture**: Uses convolutional layers for feature extraction
- ✅ **High Accuracy**: Achieves 98%+ accuracy on test data
- ✅ **Data Preprocessing**: Automatic normalization and reshaping
- ✅ **Visualization**: Displays predictions vs actual labels
- ✅ **Model Summary**: Detailed architecture overview
- ✅ **Training History**: Tracks accuracy and loss metrics
- ✅ **Clean Code**: Well-commented and organized

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- **Training Images**: 60,000 samples
- **Test Images**: 10,000 samples
- **Image Size**: 28x28 pixels (grayscale)
- **Classes**: 10 digits (0-9)
- **Format**: Pixel values ranging from 0-255

The dataset is automatically downloaded using TensorFlow's built-in dataset loader.

## 🏗️ Model Architecture

```
Input Layer (28x28x1)
       ↓
Conv2D (32 filters, 3x3) + ReLU
       ↓
MaxPooling2D (2x2)
       ↓
Conv2D (64 filters, 3x3) + ReLU
       ↓
MaxPooling2D (2x2)
       ↓
Flatten
       ↓
Dense (128 neurons) + ReLU
       ↓
Dense (10 neurons) + Softmax
       ↓
Output (Digit 0-9)
```

### Layer Details

| Layer Type | Output Shape | Parameters | Purpose |
|------------|--------------|------------|---------|
| Conv2D | (26, 26, 32) | 320 | Feature extraction |
| MaxPooling2D | (13, 13, 32) | 0 | Dimensionality reduction |
| Conv2D | (11, 11, 64) | 18,496 | Complex pattern recognition |
| MaxPooling2D | (5, 5, 64) | 0 | Spatial downsampling |
| Flatten | 1,600 | 0 | Convert to 1D |
| Dense | 128 | 204,928 | Feature learning |
| Dense | 10 | 1,290 | Classification |

**Total Parameters**: ~225,000

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### Step 2: Install Dependencies
```bash
pip install tensorflow numpy matplotlib
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Script
```bash
python mnist_cnn.py
```

## 🚀 Usage

### Basic Usage
Simply run the main script:
```bash
python mnist_cnn.py
```

### Expected Output
```
Downloading data from https://storage.googleapis.com/tensorflow/...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 26, 26, 32)        320       
maxpooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
...
=================================================================

Epoch 1/5
1875/1875 [==============================] - 45s - loss: 0.1532 - accuracy: 0.9534
...
Epoch 5/5
1875/1875 [==============================] - 43s - loss: 0.0187 - accuracy: 0.9941

Test Accuracy: 98.75%
```

### Visualization
The script displays a matplotlib window showing:
- First 5 test images
- Predicted labels
- Actual labels
- Grayscale representation of digits

## 📈 Results

### Training Performance
- **Training Accuracy**: ~99.4%
- **Validation Accuracy**: ~98.7%
- **Training Time**: ~3-5 minutes (on CPU)
- **Test Loss**: ~0.04

### Sample Predictions
| Image | Predicted | Actual | Correct |
|-------|-----------|--------|---------|
| 🖼️ | 7 | 7 | ✅ |
| 🖼️ | 2 | 2 | ✅ |
| 🖼️ | 1 | 1 | ✅ |
| 🖼️ | 0 | 0 | ✅ |
| 🖼️ | 4 | 4 | ✅ |

## 📁 Project Structure

```
mnist-digit-recognition/
│
├── mnist_cnn.py              # Main script with CNN implementation
├── README.md                 # Project documentation (this file)
└── LICENSE                   # MIT License file
```

Simple! Bas teen files hain abhi. Clean aur professional setup.

## 🎓 Learning Points

This project demonstrates:

1. **Data Preprocessing**
   - Normalization of pixel values
   - Reshaping for CNN input
   - One-hot encoding of labels

2. **CNN Architecture**
   - Convolutional layers for feature extraction
   - Pooling layers for dimensionality reduction
   - Dense layers for classification

3. **Model Training**
   - Compilation with optimizer and loss function
   - Training with validation data
   - Evaluation on test set

4. **Visualization**
   - Matplotlib for displaying results
   - Understanding model predictions

## 🔮 Future Enhancements

- [ ] Add data augmentation for better generalization
- [ ] Implement dropout layers to prevent overfitting
- [ ] Create a web interface for real-time digit recognition
- [ ] Export model for mobile deployment (TensorFlow Lite)
- [ ] Add confusion matrix visualization
- [ ] Implement model checkpointing
- [ ] Create API endpoint for predictions
- [ ] Add support for custom digit images

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

---

⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ and Python**
