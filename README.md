# CIFAR-100 Image Classification

This project implements a Convolutional Neural Network (CNN) to classify images in the CIFAR-100 dataset. The CIFAR-100 dataset consists of 100 classes, each containing 600 images of size 32x32 pixels, which makes it a challenging and widely used benchmark in image classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

---

## Introduction

This project utilizes TensorFlow/Keras to build and train a CNN for fine-grained classification of the CIFAR-100 dataset. The pipeline includes:
1. Data extraction and preprocessing
2. Model architecture definition
3. Training and evaluation
4. Visualization of performance metrics

---

## Dataset

The CIFAR-100 dataset contains:
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- Each image is labeled with one of 100 fine-grained classes.

Dataset URL: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Project Structure

```
CIFAR-100/
│
├── data/                   # CIFAR-100 dataset files
├── models/                 # Saved models and checkpoints
├── scripts/                # Scripts for training and evaluation
├── results/                # Training and evaluation results
├── README.md               # Project documentation
└── main.py                 # Main execution script
```

---

## Requirements

To run this project, install the following dependencies:

- Python 3.8+
- TensorFlow/Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model Architecture

The CNN architecture includes:
1. **Three Convolutional Layers**:
   - Filters: 32, 64, 128
   - Kernel Size: 3x3
   - Activation: ReLU
   - MaxPooling after each layer
   - Dropout to prevent overfitting

2. **Fully Connected Layers**:
   - Dense Layer (128 units, ReLU)
   - Output Layer (100 units, Softmax)

3. **Loss Function**: Categorical Crossentropy  
4. **Optimizer**: Adam with a learning rate of 0.001  
5. **Metrics**: Accuracy  

---

## Training

The training process:
1. Split the training data into sub-training (80%) and validation (20%) sets.
2. Apply one-hot encoding to labels.
3. Train the model using the Adam optimizer with the following hyperparameters:
   - Epochs: 30
   - Batch Size: 64
   - Learning Rate: 0.001
4. Visualize the accuracy and loss trends during training.

To train the model, execute:

```bash
python main.py
```

---

## Evaluation

The trained model is evaluated on the test set. Key performance metrics include:
1. **Confusion Matrix**: Highlights misclassifications across classes.
2. **Classification Report**: Includes precision, recall, and F1-score for each class.

Example code snippet for evaluation:

```python
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

---

## Results

- **Training Accuracy**: ~32%
- **Validation Accuracy**: ~38%
- **Test Accuracy**: ~38%

The results demonstrate the model's ability to classify fine-grained categories in a complex dataset. Further improvements can be achieved by:
- Fine-tuning hyperparameters
- Data augmentation
- Using advanced architectures like ResNet or DenseNet

---

## Usage

1. **Extract the CIFAR-100 Dataset**: Place the tar file in the `data/` directory.
2. **Run the Script**: Execute `main.py` to train and evaluate the model.
3. **Visualize Metrics**: Training/validation accuracy and loss will be plotted.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out for questions or contributions.
