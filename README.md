# Credit Card Fraud Detection

Kaggle dataset: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data

A machine learning project that classifies fraudulent credit card transactions using Logistic Regression and Support Vector Machine (SVM) models.

## Overview

This project addresses the challenge of detecting fraudulent transactions in highly imbalanced credit card data. By implementing classification algorithms with class balancing techniques, the models identify fraudulent patterns while managing the trade-off between catching fraud and minimizing false positives.

## Project Goals

Compared the performance of **Logistic Regression** and **Support Vector Machines (SVM)** for fraud classification to understand their strengths and implementation differences in handling imbalanced data.

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **ML Models**: Logistic Regression, SVM (Linear Kernel)
- **Preprocessing**: OneHotEncoder, MinMaxScaler, Stratified Sampling

## Dataset Features

The model uses the following key features:
- `intended_balcon_amount`: Transaction amount
- `velocity_6h`: Transaction frequency in 6-hour window
- `payment_type`: Method of payment (one-hot encoded)
- `foreign_request`: Whether transaction is foreign (binary)
- `credit_risk_score`: User's credit risk assessment
- `fraud_bool`: Target variable (0 = legitimate, 1 = fraudulent)

**Class Distribution**: 98.9% legitimate transactions, 1.1% fraudulent transactions

## Methodology

### 1. Data Exploration
- Analyzed feature correlations using heatmap visualization
- Examined severe class imbalance (197,794 legitimate vs 2,206 fraudulent)
- Identified key patterns in fraudulent vs. legitimate transactions

### 2. Data Preprocessing
- **One-Hot Encoding**: Converted categorical `payment_type` variable
- **Feature Scaling**: Applied MinMaxScaler for normalization
- **Train-Test Split**: 80-20 split with stratification to maintain class balance

### 3. Model Training & Class Imbalance Handling

#### Initial Challenge
Both models initially achieved 98.9% accuracy but caught **0% of fraud cases** - they simply predicted everything as legitimate. This revealed a critical lesson: high accuracy is meaningless with imbalanced data.

#### Solution: Class Weight Balancing
Implemented `class_weight='balanced'` parameter to penalize misclassifying the minority class (fraud) more heavily.

#### Logistic Regression
```python
log_reg = LogisticRegression(class_weight='balanced')
```
- Trained on full dataset (800,000 samples)
- Evaluated feature importance through coefficient analysis

#### Support Vector Machine (SVM)
```python
svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
```
- Trained on subset (1,000 samples) due to computational constraints
- Linear kernel for interpretability

## Results

### Model Performance Comparison

| Metric | Logistic Regression | SVM |
|--------|-------------------|-----|
| **Test Accuracy** | 65.4% | 52.2% |
| **Fraud Recall** | **64%** | 60% |
| **Frauds Detected** | 1,403 / 2,206 | 1,323 / 2,206 |
| **False Positives** | **68,414** | 94,799 |
| **Fraud Precision** | 2% | 1% |

### Logistic Regression - Confusion Matrix
```
                Predicted
              Non-Fraud  Fraud
Actual Non-F   129,380  68,414
       Fraud       803   1,403
```

### SVM - Confusion Matrix
```
                Predicted
              Non-Fraud  Fraud
Actual Non-F   102,995  94,799
       Fraud       883   1,323
```

### Winner: Logistic Regression ✓

**Logistic Regression outperformed SVM with:**
- Higher fraud detection rate (64% vs 60%)
- Fewer false positives (68,414 vs 94,799)
- Better overall accuracy (65.4% vs 52.2%)
- Ability to train on full dataset vs limited subset

## Key Insights

### The Accuracy Paradox
- Initial models achieved 98.9% accuracy but were completely useless
- They predicted everything as non-fraud, matching the class distribution
- **Learning**: Accuracy alone is a poor metric for imbalanced datasets

### Precision-Recall Trade-off
- With `class_weight='balanced'`, fraud recall improved from 0% to 64%
- Trade-off: Precision dropped to 2% (many false positives)
- In fraud detection, catching actual fraud often justifies higher false positive rates
- Business context determines acceptable thresholds

## What I Learned

- **Class imbalance is critical**: Discovered that high accuracy can be misleading with imbalanced data
- **Implemented class balancing**: Applied `class_weight='balanced'` to prioritize fraud detection
- **Model comparison**: Logistic Regression outperformed SVM in this use case
- **Feature engineering**: Transformed categorical variables using one-hot encoding and normalization
- **Evaluation metrics matter**: Understood that recall and precision are more important than accuracy for fraud detection
- **Business context**: Recognized that false positives have real costs (customer frustration) vs false negatives (financial loss)
- **Computational considerations**: Adapted SVM training approach when facing resource constraints

## Visualizations

The project includes several key visualizations:
1. **Correlation Heatmap**: Shows relationships between numeric features
2. **Class Distribution Plot**: Displays severe imbalance between fraud and non-fraud cases
3. **Confusion Matrices**: Separate visualizations for Logistic Regression and SVM performance
4. **Feature Importance**: Bar chart of logistic regression coefficients identifying top fraud indicators

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

## Future Improvements

- Implement SMOTE (Synthetic Minority Over-sampling Technique) for better class balance
- Experiment with ensemble methods (Random Forest, XGBoost) which often perform better on imbalanced data
- Perform hyperparameter tuning using GridSearchCV
- Add cross-validation for more robust model evaluation
- Implement cost-sensitive learning to account for business impact of false positives vs false negatives
- Adjust decision threshold to optimize precision-recall trade-off
- Deploy model as a REST API for real-time fraud detection
