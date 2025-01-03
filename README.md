# LoanGuard: Predicting Loan Eligibility Using Machine Learning

## Overview
**LoanGuard** is a machine learning-based loan eligibility prediction system that helps in determining the likelihood of loan approval for an individual. The project utilizes a variety of machine learning algorithms to analyze and predict whether a borrower qualifies for a loan based on historical data. The system is designed to automate and streamline the loan approval process by leveraging data science and predictive modeling techniques.

Additionally, a user-friendly **UI** has been created where users can input their details, and the model will predict whether they are eligible for the loan or not.

## Key Features
- 🧑‍💻 Data cleaning and preprocessing for enhanced model performance.
- 🔍 Exploratory Data Analysis (EDA) for insights into the dataset.
- ⚙️ Feature engineering to create more predictive features for the model.
- 📊 Performance evaluation using multiple machine learning models.
- 🚀 Deployment-ready for easy integration into financial systems.
- 🎨 **User Interface** that allows users to input personal details and receive loan eligibility predictions in real-time.

## Technologies Used
- 🐍 **Python** for data manipulation and modeling.
- 📚 **Pandas** and **NumPy** for data cleaning and processing.
- 🧠 **Scikit-learn** for machine learning models (Random Forest, Logistic Regression, KNN, Decision Tree).
- 🚀 **XGBoost** for gradient boosting-based model.
- 🧑‍🔬 **TensorFlow/Keras** for Neural Network model.
- 🖥️ **Flask** or **Streamlit** (or whatever framework you used) for creating the interactive user interface.
  
## Dataset
- 📂 The dataset used in this project is a collection of loan application data, containing various attributes related to the applicant's credit score, income, employment status, and loan application details. The goal is to predict whether the applicant will be approved for the loan or not.
- 🔗 Dataset Link: [Loan Prediction DataSet](https://www.gigasheet.com/sample-data/loan-default-prediction-dataset)

## Data Preprocessing and Feature Engineering
- 🧹 **Data Cleaning**: Removed missing or inconsistent values, and handled categorical data through encoding techniques.
- 📊 **Exploratory Data Analysis (EDA)**: Visualized key insights using plots and charts to understand correlations and distributions.
- 🔧 **Feature Engineering**: Created new features to improve model predictions, such as loan-to-income ratio and applicant’s credit history score.

## Models Used
The following machine learning algorithms were applied to the dataset for prediction:

1. **Random Forest**:
   - Accuracy: 83.16%
   - Precision: 81.09%
   - F1 Score: 83.74%
   - Recall: 86.58%

2. **Logistic Regression**:
   - Accuracy: 69.00%
   - Precision: 68.79%
   - F1 Score: 69.29%
   - Recall: 69.80%

3. **K-Nearest Neighbors (KNN)**:
   - Accuracy: 77.45%
   - Precision: 72.66%
   - F1 Score: 79.66%
   - Recall: 88.14%

4. **Decision Tree**:
   - Accuracy: 79.57%
   - Precision: 78.72%
   - F1 Score: 79.92%
   - Recall: 81.17%

5. **XGBoost**:
   - Accuracy: 86.37%
   - Precision: 90.69%
   - F1 Score: 85.63%
   - Recall: 81.11%

6. **Neural Network**:
   - Accuracy: 67.63%
   - Precision: 63.21%
   - F1 Score: 72.37%
   - Recall: 84.64%

## Performance Comparison
The table below shows the comparison of performance metrics for each model:

| Classifier           | Accuracy | Precision | F1 Score | Recall  |
|----------------------|----------|-----------|----------|---------|
| **Random Forest**     | 0.8316   | 0.8109    | 0.8374   | 0.8658  |
| **Logistic Regression** | 0.6900 | 0.6879    | 0.6929   | 0.6980  |
| **KNN**               | 0.7745   | 0.7266    | 0.7966   | 0.8814  |
| **Decision Tree**     | 0.7957   | 0.7872    | 0.7992   | 0.8117  |
| **XGBoost**           | 0.8637   | 0.9069    | 0.8563   | 0.8111  |
| **Neural Network**    | 0.6763   | 0.6321    | 0.7237   | 0.8464  |

### Best Performing Model:
- **XGBoost** achieved the highest accuracy of **86.37%** and performed well across precision, F1 score, and recall metrics. It is the best model for this particular task based on the evaluation metrics.

### Results:
![output](https://github.com/user-attachments/assets/18bbbce3-a7e4-4cde-9314-7b38f533367c)

## User Interface
The **LoanGuard** project also includes an interactive user interface (UI) where users can enter their personal details such as income, credit score, employment status, etc. Based on this input, the model will predict whether they are eligible for a loan.

- The UI can be easily accessed and run through a web browser.
- It provides an intuitive way for users to interact with the model and get predictions in real-time.
![Screenshot 2025-01-02 192730](https://github.com/user-attachments/assets/1a2b093d-fe10-496a-9eb3-e6f84d0ad1f7)

![Screenshot 2025-01-02 192755](https://github.com/user-attachments/assets/e224b808-b9c2-4a92-a5c8-336386e55f12)
