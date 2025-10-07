# Credit-Fraud-Classification

I used kaggle dataset for this project: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data

The models used were Logistic Regression and Support Vector Machines. This project was to compare and contrast the accuracy of these models when classifiying which data was fraud and how I could improve them. 

# Credit Card Fraud Detection

A machine learning project that classifies fraudulent credit card transactions using Logistic Regression and Support Vector Machine (SVM) models.

## Overview

This project addresses the challenge of detecting fraudulent transactions in imbalanced credit card data. By implementing classification algorithms and data preprocessing techniques, the model identifies fraudulent patterns while minimizing false positives.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and normalizes features
- **Exploratory Data Analysis**: Correlation heatmaps and class distribution visualization
- **Multiple Models**: Implements both Logistic Regression and SVM classifiers
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **Feature Importance**: Visualizes coefficient weights to identify key fraud indicators

## Technologies Used

- **Python 3.x**
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **OneHotEncoder**: Categorical variable encoding
- **MinMaxScaler**: Feature normalization

## Dataset Features

The model uses the following key features:
- `intended_balcon_amount`: Transaction amount
- `velocity_6h`: Transaction frequency in 6-hour window
- `payment_type`: Method of payment (one-hot encoded)
- `foreign_request`: Whether transaction is foreign (binary)
- `credit_risk_score`: User's credit risk assessment
- `fraud_bool`: Target variable (0 = legitimate, 1 = fraudulent)

## Methodology

### 1. Data Exploration
- Analyzed feature correlations using heatmap visualization
- Examined class imbalance in fraud distribution
- Identified key patterns in fraudulent vs. legitimate transactions

### 2. Data Preprocessing
- **One-Hot Encoding**: Converted categorical `payment_type` variable
- **Feature Scaling**: Applied MinMaxScaler for normalization
- **Train-Test Split**: 80-20 split with stratification to maintain class balance

### 3. Model Training

#### Logistic Regression
- Binary classification with default hyperparameters
- Trained on full dataset
- Evaluated feature importance through coefficient analysis

#### Support Vector Machine (SVM)
- Linear kernel with C=1.0
- Trained on subset (1000 samples) for computational efficiency
- Evaluated on full test set

### 4. Model Evaluation
- **Confusion Matrix**: Visual representation of true positives, false positives, true negatives, and false negatives
- **Classification Report**: Precision, recall, and F1-score for both classes
- **Accuracy Metrics**: Training and testing accuracy for model validation

## Results

Both models were evaluated on their ability to correctly classify fraudulent transactions while minimizing false alarms. Key metrics include:

- Training and testing accuracy scores
- Precision and recall for fraud detection
- Confusion matrix analysis showing prediction distributions
- Feature coefficient visualization identifying top fraud indicators

## Visualizations

The project includes several key visualizations:
1. **Correlation Heatmap**: Shows relationships between numeric features
2. **Class Distribution**: Displays imbalance between fraud and non-fraud cases
3. **Confusion Matrices**: Separate visualizations for Logistic Regression and SVM performance
4. **Feature Importance**: Bar chart of logistic regression coefficients

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git

# Navigate to project directory
cd credit-card-fraud-detection

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

```python
# Run the Jupyter notebook or Python script
jupyter notebook fraud_detection.ipynb
```

## Project Goals

Compared the performance of **Logistic Regression** and **Support Vector Machines (SVM)** for fraud classification to understand their strengths and implementation differences in handling imbalanced data.

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **ML Models**: Logistic Regression, SVM
- **Preprocessing**: OneHotEncoder, MinMaxScaler

## What I Learned

- How to implement and compare multiple classification algorithms
- Handling imbalanced datasets with stratified sampling
- Feature engineering through one-hot encoding and normalization
- Evaluating models using precision, recall, F1-score, and confusion matrices
- Interpreting model performance in business context (false positives vs false negatives)

## Future Improvements

- Implement oversampling techniques (SMOTE) to handle class imbalance
- Experiment with ensemble methods (Random Forest, XGBoost)
- Perform hyperparameter tuning using GridSearchCV
- Add cross-validation for more robust model evaluation
- Deploy model as a REST API for real-time fraud detection
- Implement cost-sensitive learning to account for business impact of false positives vs false negatives

