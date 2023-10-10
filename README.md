# Loan-Status-Prediction
This Python script demonstrates how to predict loan status using machine learning techniques.
The dataset used for this project is loaded from a CSV file, cleaned, and then used to train an SVM (Support Vector Machine) classifier. 
The trained model achieves an accuracy of 78% on the test data.

## Usage
Clone this repository or download the loan_status_prediction.py script.
Download the dataset (train.csv) and place it in the same directory as the script.
Run the script using the following command: python loan_status_prediction.py

## Description
### Data Loading and Exploration: The script starts by loading the dataset (train.csv) using Pandas and displays the first few rows, the dataset's shape, and statistical information.
### Data Cleaning: It checks for missing values and handles them by dropping rows with missing data. You can choose between imputing missing values with column means or dropping them, as commented in the code.
### Data Preprocessing: Textual columns are converted to numerical using Label Encoding. You can also replace specific values in columns if needed.
### Data Visualization: Various visualizations are created to understand the data, including a heatmap of feature correlations and count plots for relationships between categorical features and loan status.
### Data Splitting: The data is split into input (X) and label (Y) data. Input data is standardized using StandardScaler to bring all features to a common range.
### Model Training: An SVM (Support Vector Machine) classifier is created and trained on the training data.
### Model Evaluation: The script calculates and displays accuracy scores for both training and testing data predictions. The model achieves an accuracy of 78% on the test data.
### Making Predictions: Finally, the script provides a predictive system where you can input your own data to predict whether a loan will be approved ("Yes" or "No").

## Contributing
the contributions are welcome and make fork the repository, make changes, and submit pull requests.

