# Alphabet Soup Charity Deep Learning Model
This project explores the use of deep learning to predict which organizations funded by Alphabet Soup Charity are likely to be successful. The goal is to create a binary classification model that will accurately predict whether an organization will be successful or not based on various features.

## Overview
Alphabet Soup Charity is a non-profit organization that provides funding for philanthropic causes around the world. The organization receives donations from various sources and then distributes the funds to other charitable organizations. In order to ensure that the funds are being used effectively, Alphabet Soup Charity wants to create a model that can predict whether an organization will be successful based on certain features.

## Data Preprocessing
The target variable for our model is the IS_SUCCESSFUL column, which indicates whether an organization was successful or not. The features for our model are all other columns except for EIN and NAME, which are identification columns that are not relevant to our analysis.

Before creating the model, we preprocessed the data by performing the following steps:

- Dropped the EIN and NAME columns
- Binned low-frequency values in the APPLICATION_TYPE and CLASSIFICATION columns into an Other category
- Grouped low-frequency values in the AFFILIATION, USE_CASE, and ORGANIZATION columns into an Other category
- Converted the INCOME_AMT column into a categorical variable and one-hot encoded the categories
- Scaled the ASK_AMT column using the MinMaxScaler function
- Converted categorical data to numeric using the pd.get_dummies function
- Compiling, Training, and Evaluating the Model
- We created a deep learning model using the Keras library. The model has 9 input neurons and 3 hidden layers, each with a variable number of neurons and activation functions selected using the Keras Tuner library. The output layer has 1 neuron with a sigmoid activation function.

We compiled the model using the binary crossentropy loss function and the RMSprop optimizer. The model was trained on a dataset split into training and testing sets using a StandardScaler to scale the data. We used the ModelCheckpoint callback function to save the weights of the model every 5 epochs during training.

The model achieved an accuracy of 72.69% on the testing set, which did not meet our target model performance of 75%. We attempted to improve the model performance by adjusting the number of neurons and layers, changing the activation functions, and adjusting the learning rate, but were not able to improve the accuracy.

## Summary
Our deep learning model was able to achieve an accuracy of 72.69% on the testing set, which did not meet our target model performance of 75%. We attempted to improve the model performance by adjusting the number of neurons and layers, changing the activation functions, and adjusting the learning rate, but were not successful in improving the accuracy.

## Recommendations
If we were to attempt to solve this classification problem using a different model, we might consider using a random forest classifier or a support vector machine (SVM). Random forest classifiers are effective for classification problems with complex relationships between variables, while SVMs are effective for datasets with a high number of features. We could also consider feature selection or engineering to reduce the number of features and improve the model's performance.

# References:
- IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/
- TensorFlow. "Getting started with the Keras Sequential model." TensorFlow. https://www.tensorflow.org/guide/keras/sequential_model
- Scikit-learn. "sklearn.model_selection.train_test_split." Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Scikit-learn. "sklearn.preprocessing.StandardScaler." Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Keras Tuner. "Tuner Documentation." Keras Tuner. https://keras-team.github.io/keras-tuner/documentation/tuners/
Brownlee, Jason. "A Gentle Introduction to Dropout for Regularizing Deep Neural Networks." Machine Learning Mastery. https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
