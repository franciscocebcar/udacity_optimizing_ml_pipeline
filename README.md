# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about marketing campaigns conducted at a financial institution, including a few features from the customers (whether they are in default, whether they have a loan, marital status, etc), and an outcome variable (y) that seems to indicate whether the campaign was successful or not.

This is a classification problem.

The project consisted in fitting a logistic regression model doing hyperparameter tuning using hyperdrive (finding optimal values for hyperparameters C and max_iter), and also using AutoML to fit many different models (not only logistic regression) and find the best performing model.

## Scikit-learn Pipeline
This part of the project consisted in using HyperDrive to train Logistic Regression models using different values of hyperparameters (C and max_iter). It does so by calling a train.py script multiple times, once for each combination of the hyperparameters.

The train.py script performs the following:
- Retrieves the data from a URL as a DataSet, pre-processes the data by generating dummies for categorical variables and converting textual variables to 0 or 1 (binarizing) and mapping months and weekdays as numbers (not sure it's a great idea since this is considering those features as ordinal).
- Splits the dataset into a training and a test dataset (80/20 split)
- Fits a logistic regression model using the parameters passed (C and max_iter), calculates the accuracy on the test dataset and logs this metric.

The HyperDrive part of the pipeline included a random parameter sampling object that would pick values from a random uniform distribution between 0.0 and 1.0 for C, and a random discrete value between 50 and 100 for max_iter. The benefit of this parameter sampler over the grid search is that it does not do an exhaustive test for all possible combinations and therefore is faster.

The bandit early stopping policy was used to stop training when deviating 20% from the best run after 5 intervals, checking after every 3.

The best performing model after 20 run was the logistic regression with C equal to 0.012214103872074666 and max_iter equal to 93. 

## AutoML
The AutoML experiment took the dataset and performed a number of preprocessing actions prior to training a variety of models on it. The best model was the VotingEnsemble, which had an accuracy of 0.9165 

## Pipeline comparison
AutoML is a very powerful feature because it tries many different models that might make different assumptions about the data. A voting ensemble model provides better predictions as it uses a diverse set of models that make different errors, hence why the VotingEnsemble gave better accuracy.

## Future work
More careful consideration of the features is something that could potentially help with the performance of the models. 

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
