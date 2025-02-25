# Neural Network Classifier with Hyperparameter Tuning

## Project Overview

This repository contains a Python implementation of a neural network classifier designed for binary classification tasks. The model is built using TensorFlow and Keras, with hyperparameter optimization performed via Keras Tuner's RandomSearch. The project includes data preprocessing, visualization, model training, evaluation, and feature importance analysis using SHAP. The final model achieves a test accuracy of 99.58% on the provided dataset.

The codebase is structured to be modular and reproducible, making it suitable for machine learning practitioners and researchers working on classification problems.

## Features

- Data preprocessing: Handling missing values, encoding categorical variables, and feature scaling.
- Data visualization: Correlation heatmaps, pair plots, and histograms for exploratory data analysis.
- Model architecture: A deep neural network with configurable layers, dropout, batch normalization, and LeakyReLU activation.
- Hyperparameter tuning: Automated tuning of layer units, dropout rates, and learning rates using RandomSearch.
- Model evaluation: Accuracy, loss plots, classification report, and confusion matrix.
- Feature importance: SHAP analysis to interpret model predictions.
- Model persistence: Saving the optimized model for future use.

## Dataset

The code assumes a CSV dataset (`diabetes_dataset.csv`) with numerical and categorical features, including an `Outcome` column as the binary target variable. Example features include `Age`, `BMI`, `Glucose`, `BloodPressure`, `FamilyHistory`, `DietType`, `Hypertension`, and `MedicationUse`. Replace `diabetes_dataset.csv` with the path to your dataset and adjust feature names as needed.

Test Accuracy: 99.58%

                   precision    recall  f1-score   support
               0       0.99      1.00      1.00      1275
               1       1.00      0.99      0.99       633
        accuracy                           1.00      1908
       macro avg       1.00      0.99      1.00      1908
    weighted avg       1.00      1.00      1.00      1908
    
  The high precision, recall, and F1-scores indicate excellent performance across both classes, with minimal misclassifications.

## Acknowledgements

- Built with TensorFlow, Keras, and Keras Tuner.
- Feature importance analysis powered by SHAP.
- Dataset preprocessing and visualization supported by Pandas, NumPy, Matplotlib, and Seaborn.
