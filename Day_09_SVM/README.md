a
# Day 09 - Support Vector Machine (SVM)

## Objective
To build a machine learning classification model using Support Vector Machine (SVM).

## Dataset
Breast Cancer Wisconsin Dataset from Scikit-learn.

- Total samples: 569
- Total features: 30
- Training samples: 455
- Testing samples: 114
- Target classes: Malignant and Benign

## Concepts Learned
- Support Vector Machine (SVM)
- Classification
- Training and testing data
- Feature scaling using StandardScaler
- Linear kernel
- Model evaluation using accuracy
- Precision, recall, and F1-score

## Algorithm
SVM is a supervised machine learning algorithm used for classification. It finds a decision boundary that separates different classes.

## Steps Performed
1. Loaded the Breast Cancer Wisconsin dataset.
2. Split the dataset into training and testing sets.
3. Scaled the features using StandardScaler.
4. Trained the SVM model using a linear kernel.
5. Predicted the test data results.
6. Evaluated the model using accuracy and a classification report.

## Model Results

- Accuracy: 97.37%
- Training samples: 455
- Testing samples: 114

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.95 | 0.98 | 0.96 | 42 |
| Benign | 0.99 | 0.97 | 0.98 | 72 |

## Libraries Used
- Scikit-learn
- Python

## How to Run

```bash
python svm.py
```

## Conclusion
The SVM model achieved 97.37% accuracy on the test dataset. This practical helped me understand SVM classification, feature scaling, and model evaluation.
