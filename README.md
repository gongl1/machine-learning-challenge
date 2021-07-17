# Machine Learning Homework - Exoplanet Exploration

##### Conclusions are at the end of this readme file. 

![exoplanets.jpg](Images/exoplanets.jpg)

### Before You Begin

1. Create a new repository for this project called `machine-learning-challenge`. **Do not add this homework to an existing repository**.

2. Clone the new repository to your computer.

3. Give each model you choose their own Jupyter notebook, **do not use more than one model per notebook.**

4. Save your best model to a file. This will be the model used to test your accuracy and used for grading.

5. Commit your Jupyter notebooks and model file and push them to GitHub.

## Note

Keep in mind that this homework is optional! However, you will gain a much greater understanding of testing and tuning different Classification models if you do complete it.

## Background

Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.

To help process this data, you will create machine learning models capable of classifying candidate exoplanets from the raw dataset.

In this homework assignment, you will need to:

1. [Preprocess the raw data](#Preprocessing)
2. [Tune the models](#Tune-Model-Parameters)
3. [Compare two or more models](#Evaluate-Model-Performance)

- - -

## Instructions

### Preprocess the Data

* Preprocess the dataset prior to fitting the model.
* Perform feature selection and remove unnecessary features.
* Use `MinMaxScaler` to scale the numerical data.
* Separate the data into training and testing data.

### Tune Model Parameters

* Use `GridSearch` to tune model parameters.
* Tune and compare at least two different classifiers.

### Reporting

* Create a README that reports a comparison of each model's performance as well as a summary about your findings and any assumptions you can make based on your model (is your model good enough to predict new exoplanets? Why or why not? What would make your model be better at predicting new exoplanets?).

- - -

## Resources

* [Exoplanet Data Source](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

* [Scikit-Learn Tutorial Part 1](https://www.youtube.com/watch?v=4PXAztQtoTg)

* [Scikit-Learn Tutorial Part 2](https://www.youtube.com/watch?v=gK43gtGh49o&t=5858s)

* [Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)

- - -

## Hints and Considerations

* Start by cleaning the data, removing unnecessary columns, and scaling the data.

* Not all variables are significant be sure to remove any insignificant variables.

* Make sure your `sklearn` package is up to date.

* Try a simple model first, and then tune the model using `GridSearch`.

* When hyper-parameter tuning, some models have parameters that depend on each other, and certain combinations will not create a valid model. Be sure to read through any warning messages and check the documentation

- - -

## Submission

* Create a Jupyter Notebook for each model and host the notebooks on GitHub.

* Create a file for your best model and push to GitHub

* Include a README.md file that summarizes your assumptions and findings.

* Submit the link to your GitHub project to Bootcamp Spot.

* Ensure your repository has regular commits (i.e. 20+ commits) and a thorough README.md file

## Rubric

[Unit 21 Rubric - Machine Learning Homework - Exoplanet Exploration](https://docs.google.com/document/d/1IcLYc8KHt82ftMcsueM6s7rn9nexuN4PqHSJDUa7e2Y/edit?usp=sharing)

- - -

##### Conclusions

# Goal

Various machine learning classification models were used to predict candidate exoplanet classifications. Grid Search was used to increase the accuracy of the model. 

# Process

**Data Cleaning and Pre-Processing**

Data was first read in from a csv file, and null columns and null rolls were dropped. After this, there were still several columns available to select as features to train the model on. In order to use the most relevant features, the top ten features of the data set ranked by feature importances were found by using `ExtraTreesClassifier()` and those top ten features were stored as a series to be used as `X` values. The `koi_disposition` column contained the classification values of each exoplanet candidate and would be used as `y` values. 

With `X` and `y` values set, data was split into training and testing sets using `train_test_split` with `stratify=y` to ensure that there was an even distribution of classification values in both data sets. Then, `MinMaxScaler` was used to scale both sets of `X` data.

This method was used for all four models.

**Logistic Regression**

I initialized the model using `LogisticRegression()` and fit the model using the training data. Model was scored using both the training and testing data. Both sets scored fairly well, with the training data at 84.3% and testing data at 82.9%.

`GridSearchCV` was used to further tune the parameters to create a better scoring model. The parameters were set to explore different `C` values using both L1 and L2 penalties as regularization methods. A new model was then fit using this grid with the newly found best parameters, before predicting on the test data. This new model's score was better than the original, scoring at 85.7%. Classification report was calculated at the end. 

**K-Nearest Neighbors**

To find the best k value to use in this model, a loop was created to run through a set of possible k values. Because there are three possible classifications, the range of k values was started at 5 with a step of 3 to avoid any even split of classifications. Comparing the training and testing scores of each model, it looked like k=29 was the best value, as it had the lowest difference between training and testing scores, without the testing score being higher than the training.

To further tune the model’s parameters, `GridSearchCV` was used and the possible values of k were expended. The model was then retrained using the best parameter found and the model was scored using the test set of data.

Grid Search found k=42 to be the best k value, with an accuracy of 85.4%, so this model was not improved by the use of Grid Search.

**Random Forest**

The model initialized using `RandomForestClassifier()` and set the number of trees to 300 (`n_estimators=300`). The model was then fit and scored, with the testing data scoring at 87.9%.

Using Grid Search, different parameters were explored including `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. Again, this grid was used to train a new model, before predicting and scoring. The new model scored at 88.8%, very close to the original model.

**Support Vector Machine**

The model was initialized with `SVC()` and the kernel was set to `linear` before training and scoring the model, with the testing data scoring at 80.5%.

With Grid Search, various C values, gamma values, and linear and rbf kernels were explored. After training the new model, accuracy increased to 84.7%.

# Summary

Overall, all models scored in the 80% range with the use of Grid Search slightly improving accuracy. Grid Search worked best on the logistic regression and SVM models, increasing accuracy by around 3% respectively. In terms of best classifying exoplanet candidates, the random forest model was most accurate at 88.8%. 
