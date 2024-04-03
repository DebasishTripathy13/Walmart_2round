# Hard Disk Failure Prediction with Random Forest Classifier

This project utilizes a Random Forest Classifier to predict hard disk failures based on SMART (Self-Monitoring, Analysis and Reporting Technology) attributes.

## Data Source

The data used in this project is sourced from Kaggle and contains SMART attributes for a large number of hard disks, along with their failure status.

## Methodology

The notebook performs the following steps:

**Data Preprocessing:**
- Upsampling the minority class (failed disks) to address class imbalance.
- Dropping irrelevant columns and handling missing values.

**Model Training:**
- Splitting the data into training and testing sets.
- Training a Random Forest Classifier on the training data.

**Model Evaluation:**
- Evaluating the model's performance on the testing data using metrics like accuracy, precision, recall, F1-score, and Matthews correlation coefficient.
- Visualizing the confusion matrix to understand the model's prediction patterns.

**Model Saving and Loading:**
- Saving the trained model using both joblib and pickle for compatibility.
- Demonstrating how to load the saved model for future predictions.

**Interactive Prediction:**
- Defining a function that takes user inputs for SMART attributes.
- Using the loaded model to predict the failure status of a hard disk based on user-provided inputs.

## Results

The Random Forest Classifier achieves an accuracy of 86.45% on the testing data, indicating good predictive performance. The confusion matrix further reveals the model's ability to correctly identify both normal and failed hard disks.

## Usage

This notebook provides a framework for hard disk failure prediction. You can use it to:
- Train and evaluate the model on different datasets.
- Fine-tune the model parameters for improved performance.
- Develop an interactive application for predicting hard disk failures based on user inputs.

## Requirements

To run this notebook, you will need the following libraries:
- pandas
- sklearn
- joblib
- pickle
- matplotlib
- seaborn

## Disclaimer

This project is for educational purposes only and should not be used for critical decision-making without further validation and testing.


# Code for running the above model for prediction
```
with open('random_forest_model_for_hard_disk.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Defining a function to take user inputs and make predictions
def predict_hard_disk_failure():
    # Ask the user for input features
    smart_5_raw = float(input("Enter the value of smart_5_raw: "))
    smart_187_raw = float(input("Enter the value of smart_187_raw: "))
    smart_188_raw = float(input("Enter the value of smart_188_raw: "))
    smart_197_raw = float(input("Enter the value of smart_197_raw: "))
    smart_198_raw = float(input("Enter the value of smart_198_raw: "))

    '''Create a DataFrame with the user inputs
        these are for the parameters that we have trained the model on based on the SMART'''
    input_data = {
        'smart_5_raw': [smart_5_raw],
        'smart_187_raw': [smart_187_raw],
        'smart_188_raw': [smart_188_raw],
        'smart_197_raw': [smart_197_raw],
        'smart_198_raw': [smart_198_raw]
    }
    input_df = pd.DataFrame(input_data)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_df)[0]

    # Display the prediction
    if prediction == 0:
        print("The hard disk is predicted to be normal.")
    else:
        print("The hard disk is predicted to fail.")

# Call the function to make predictions based on user inputs  Implementation
predict_hard_disk_failure()
```
