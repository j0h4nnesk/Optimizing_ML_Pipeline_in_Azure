# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project I built **Azure ML pipeline** using **Python SDK** and a custom **Scikit-learn Logistic Regression** model. Hyperparameters of the model were optimized using Hyperdrive. After this Azure AutoML was used to find optimal model using the same dataset, and the **results of the two methods were compared**. 

See the diagram below showing the main steps followed in the project:

![alt text](images/Project_process_overview.png)

**Step 1:** Setting up the training [script](train.py), to create a Tabular Dataset from imported file, clean and split the data for Scikit-learn logistic regression model. 

**Step 2:** Creating [Jupyter Notebook](udacity-project.ipynb) and configuring HyperDrive to find the best hyperparameters for the logistic regression model. 

**Step 3:** Loading the same dataset, as used with Scikit-learn model, with TabularDatasetFactory and using AutoML to find an optimized model. 

**Step 4:** Comparing the results of the two methods and writing a research summary, this RAEADME.md.

## Summary
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y). 

The best performing model with **0.91627 accuracy** was the **AutoML model** which used **VotingEnsemble** algorith. The best **HyperDrive model** using **Scikit-learn logistic regression** had nearly as good accuracy of **0.90895**.

## Scikit-learn Pipeline

Built Scikit-learn pipeline uses [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model for classification, while the hyperparameters were tuned using [HyperDrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters). The used hyperparameters were C (Inverse of regularization strength) with unform value distribution and max_iter (Maximum number of iterations taken for the solvers to converge) with discrete values for choice. 

**Parameter Sampling**

RandomParameterSampling, where hyperparameter values are randomly selected from the defined search space, was used as a sampler. It is a good choice as it is [more efficient, though less exhaustive compared](https://www.sciencedirect.com/science/article/pii/S1674862X19300047) to Grid grid search to search over the search space.

```
ps = RandomParameterSampling({
    '--max_iter' : choice(20,40,80,100,150,200),
    '--C' : uniform(0.001,10)
}) 
```
**Early Stopping Policy**

Early stopping policy was used to terminate poorly performing runs, this also improves computational efficiency. Here [**Bandit policy**](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters) was used as an early stopping policy, and was configured as follows:
```
policy = BanditPolicy(evaluation_interval=2,slack_factor=0.1,delay_evaluation=1)
```
In this example, the early termination policy is applied at every second interval when metrics are reported, starting at evaluation interval 1. Any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be terminated.

## AutoML

[AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) works in a way that it creates a number of pipelines in parallel that tries different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment.

Here I chose the following confifuration for the AutoML run:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    n_cross_validations=2)
```
Where:

*experiment_timeout_minutes* = Maximum amount of time in hours that all iterations combined can take before the experiment terminates. 

*task* = Defines what kind of problem to solve, here 'classification' was chosen

*primary_metric* = The metric that Automated Machine Learning will optimize for model selection, here 'accuracy' was chosen

*n_cross_validations* = [This parameter sets how many cross validations to perform, based on the same number of folds](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits) (folds=subsections). [Cross validation is a technique that allows us to produce test set like scoring metrics using the training set. It simulates the effects of “going out of sample” using just our training data, so we can get a sense of how well our model generalizes.](https://towardsdatascience.com/understanding-cross-validation-419dbd47e9bd) Chosen value '2' means, two different trainings, each training using 1/2 of the data, and each validation using 1/2 of the data with a different holdout fold each time. As a result, metrics are calculated with the average of the two validation metrics.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
