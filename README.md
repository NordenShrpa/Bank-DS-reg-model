# Bank-DS-reg-model
Bank dataset regression model using Python (tensorflow/ keras)

# Bank Dataset Regression Model

## Overview
This project is a machine learning regression model developed using **TensorFlow/Keras** and the **Bank Dataset**. The model predicts the estimated salary of customers based on various features such as age, geography, tenure, and balance. The dataset has been preprocessed, and the model has been trained using dense neural network layers.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Technologies Used](#technologies-used)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Details](#training-details)
6. [Evaluation Metrics](#evaluation-metrics)
7. [How to Run](#how-to-run)
8. [Future Improvements](#future-improvements)

---

## Dataset
The dataset used for this project contains customer data from a bank. 
- **Features**:
  - Age
  - Credit Score
  - Geography
  - Gender
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Exited (Target)

- **Target Variable**:
  - `EstimatedSalary` (continuous variable)

**Source**: Hypothetical dataset used for machine learning practice.

---

## Technologies Used
The following technologies and libraries were used in the project:
- **Python** (Programming Language)
- **TensorFlow/Keras** (Deep Learning Library)
- **Scikit-learn** (Machine Learning Tools)
- **Pandas** (Data Manipulation)
- **NumPy** (Numerical Computations)
- **Matplotlib/Seaborn** (Data Visualization)

---

## Preprocessing
1. Dropped unnecessary columns: `RowNumber`, `CustomerId`, and `Surname`.
2. Categorical encoding of the `Geography` and `Gender` features using **LabelEncoder**.
3. Split the dataset into training and testing sets (default 80:20 split).
4. The target variable for this project is continuous, so the model performs regression.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

y = data["EstimatedSalary"]
X = data.drop(["EstimatedSalary"], axis=1)

# Encoding categorical features
geography_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
X["Geography"] = geography_encoder.fit_transform(X["Geography"])
X["Gender"] = gender_encoder.fit_transform(X["Gender"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## Model Architecture
The model is a fully connected **Dense Neural Network** built using TensorFlow/Keras. The architecture includes:
- Input Layer: 10 features
- Hidden Layers:
  - Dense (10 units, ReLU activation)
  - Dense (20 units, ReLU activation)
  - Dense (20 units, ReLU activation)
  - Dense (10 units, ReLU activation)
- Output Layer:
  - Dense (1 unit, Linear activation for continuous predictions)
  - Can also use sigmoid or ReLU activation function for the output layer.


### Model Compilation
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)

---

## Training Details
- **Epochs**: 25
- **Validation Split**: 10%
- **Batch Size**: Default (32)

```python
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train, y_train, validation_split=0.1, epochs=25)
```

---

## Evaluation Metrics
Since this is a regression problem, the following metrics were used:
1. **Mean Squared Error (MSE)**
2. **Mean Absolute Error (MAE)**
3. **R² Score** (Coefficient of Determination)

---

## Future Improvements
- Hyperparameter tuning using techniques like Grid Search or Random Search.
- Experiment with dropout layers to reduce overfitting.
- Test other activation functions like `sigmoid` or `tanh` in intermediate layers.
- Convert the regression problem into a binary classification problem (e.g., high vs low salary prediction).

---
