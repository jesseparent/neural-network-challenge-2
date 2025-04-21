
# Neural Network Challenge 2: Employee Attrition and Department Prediction

This repository contains my solution to the **Module 19 Challenge**, where I built a **branched neural network** to help HR predict:

1. Whether an employee is likely to leave the company (`Attrition`)
2. Which department may be the best fit for each employee (`Department`)

---

## Background

HR wants to improve retention and employee placement by using machine learning. The goal of this challenge is to develop a **multi-output neural network** that can:
- Predict **attrition** (binary classification)
- Predict **department reassignment** (multi-class classification)

---

## Files

- `attrition.ipynb` – Jupyter Notebook containing data preprocessing, model building, training, and evaluation.
- Processed features and model metrics are included in the notebook output.

---

## Model Workflow

### Part 1: Preprocessing
- Imported and reviewed employee dataset
- Created:
  - `X_df` using 10 selected features
  - `y_df` with `Attrition` and `Department`
- Encoded all categorical columns using `OneHotEncoder`
- Scaled the features using `StandardScaler`
- Split the dataset into training and testing sets

### Part 2: Model Creation & Training
- Built a **branched neural network** using Keras functional API:
  - Shared hidden layers: 2 layers
  - Branch 1 (Attrition): 1 hidden layer + 1 sigmoid output node
  - Branch 2 (Department): 1 hidden layer + softmax output layer
- Compiled with:
  - `adam` optimizer
  - `binary_crossentropy` and `categorical_crossentropy` losses
- Trained the model on the scaled data
- Evaluated model performance on test set

### Part 3: Evaluation & Summary
- **Attrition Accuracy:** ~82.2%
- **Department Accuracy:** ~58.5%
- Model loss and output losses are printed in the notebook

---

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/jesseparent/neural-network-challenge-2.git
   ```

2. Open `attrition.ipynb` in Jupyter or Google Colab.

3. Install dependencies:
   ```bash
   pip install tensorflow pandas scikit-learn
   ```

4. Run all cells to reproduce the model training and evaluation.

---

