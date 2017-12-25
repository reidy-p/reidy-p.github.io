---
layout: post
author_profile: true
title: "Estimating Trump's Fake Twitter Followers: Part 2"
output: html_document
date: 2017-12-24 12:45:00 -0400
---

In the previous post I collected some data on a sample of Donald Trump's Twitter followers and also assembled a training dataset. In this post I will try to use some simple machine learning models to estimate how many fake followers Trump may have. I will start by training and testing various machine learning models on the Varol et al. (2017) [dataset](https://botometer.iuni.iu.edu/bot-repository/datasets.html). I will then use the trained models to classify bots in the Trump followers dataset that I have collected (which has no labels).

Before fitting any models it's important to look at the features that we will use to train the models and make sure that there are no obvious errors. We will also need to do some preprocessing to make sure that the features will work with the models:


```python
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import OneHotEncoder
plt.style.use('fivethirtyeight')
%matplotlib inline
rcParams['figure.figsize'] = (10, 10)

# Load data
training_df = pd.read_csv('Data/varol2017_data/varol2017_training_data.csv', parse_dates=['created_at'])

# Convert boolean columns to int
bool_cols = training_df.select_dtypes(include=[bool]).columns
training_df.loc[:, bool_cols] = training_df.loc[:, bool_cols].astype(int)

# Cleaning up some features
training_df['followers_friends_ratio'] = ((training_df['followers_count'] / training_df['friends_count'])
                                          .replace(np.inf, np.nan)) # remove any infinity values
training_df['description_len'] = training_df['description'].str.len().replace(np.nan, 0) 

# List of features
training_df.columns
```

<img src="/assets/trump_fakeaccounts_part2_files/figure1.png"/>


I will exclude features that we would expect to have little predictive value or that would be difficult to interpret. For example, ``account_id`` should not be able to predict whether an account is a bot and, even if it did, it's not clear how we would interpret it (i.e., what does a one unit change in ``account_id`` mean?). I therefore exclude ``account_id``, ``name``, ``screen_name`` and ``description`` as features. If we wanted to use these features in the models we would have to transform them in some way so that they would make sense in the context of a model. For example, we already have a ``description_len`` feature which is the length in characters of the description provided on the Twitter account. This makes sense because it is a continuous value so we can use it in most machine learning models. I also drop the ``lang`` column from the dataset because all the accounts in the dataset are in English.


```python
training_df = training_df.drop(labels=['account_id', 'name', 'screen_name', 'description', 'lang'], axis=1)
```


There are some other features that we should transform before fitting any models. For example, a feature with the account creation date (e.g., 2015-03-19 22:14:20) may not make sense. But we can use this feature to calculate the account age in numbers of days which should work better. 


```python
training_df['account_age_days'] = (datetime.datetime(2017, 12, 18) - training_df['created_at']).dt.days
training_df = training_df.drop(labels=['created_at'], axis=1)
training_df['account_age_days'].head()
```
<img src="/assets/trump_fakeaccounts_part2_files/figure2.png"/>

Next we can take a look at any categorical variables and ensure that they are in the correct format. The first step is to check that all categorical variables have been converted from strings to integer values.


```python
training_df[['bot', 'default_profile', 'default_profile_image', 'geo_enabled', 'location', 'protected', 
             'time_zone', 'verified']].head()
```
<img src="/assets/trump_fakeaccounts_part2_files/figure3.png"/>


``bot``, ``default_profile``, ``default_profile_image``, ``geo_enabled``, ``protected``, and ``verified`` are all binary variables and have already been encoded as 0 (no) or 1 (yes).

``location`` and ``time_zone`` have not been transformed to integer values. These variables have more than two possible categories. We could transform them in a similar way to the binary variables. For example, if there were five categories in ``location`` we could replace each value in the column with a value in [0, 1, 2, 3, 4] depending on the category. But the problem with this is that it implies that the "distance" between each of the categories is the same and that the order of the categories has some meaning. These are generally not good assumptions so instead we should use [One-Hot Encoding](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) or [pd.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html). In this method we create a new column for each category in an existing column and create a binary value for this category. For example, if the values of location are 'USA', 'England', and 'France' then we would have three new columns and if a new observation has a value of 'USA' we would have [1, 0, 0].

However, the ``location`` feature has a very large number of values and many of them overlap because the account owner can enter this description manually rather than choosing from a pre-defined set of locations. For example, many accounts have "United States" as the location while others have "USA" or the name of a US city of US state. If we wanted to use the column as a feature in the models we would have to try to combine a lot of the unique values. In addition, the account owner can enter any value that they wish in this field so it may not be reliable even if we manage to clean it up. For these reasons I will drop this column from the analysis.


```python
training_df['location'].nunique()
```

1248

```python
training_df['location'].value_counts()[:10]
```
<img src="/assets/trump_fakeaccounts_part2_files/figure4.png"/>


```python
training_df = training_df.drop(labels=['location'], axis=1)
```

The number of unique values for the ``time_zone`` column is much smaller than location but it's not clear if time-zone would be a good predictor of bot status so I will exclude it for now.


```python
training_df['time_zone'].nunique()
training_df = training_df.drop(labels=['time_zone'], axis=1)
```

After this pre-processing we are left with the following features:


```python
training_df.columns
```
<img src="/assets/trump_fakeaccounts_part2_files/figure5.png"/>


## Useful Diagnostic Functions


```python
def draw_roc_curve(y_val, model_probs_target, ax, fold_num=None):
    """Draw the ROC curve given the true y values (y_val) and 
    the probability estimates of the target class (model_probs_val)"""
    
    fpr, tpr, thresholds = roc_curve(y_val, model_probs_target)  
    roc_auc = auc(fpr, tpr)
    
    if fold_num:
        ax.plot(fpr, tpr, label='ROC fold {} (AUC = {:.2f})'.format(fold_num, roc_auc))
    else:
        ax.plot(fpr, tpr, label=('AUC = {:.2f}'.format(roc_auc)))
    
    ax.set_title('ROC Curve')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc=0, fontsize='medium')

def draw_learning_curve(estimator, X_data, y_data, ax, cv=5):
    """Draw the learning curve given the instantiated estimator, the full matrix of 
    features (X_data) and the response variable (y_data) for the training data"""
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X=X_data, 
                                                             y=y_data, cv=cv, shuffle=True, 
                                                             random_state=4, 
                                                             train_sizes=list(range(10, 1250, 15)))
    ax.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
    ax.plot(train_sizes, valid_scores.mean(axis=1), label='Cross-Validation Score')
    ax.set_title("Learning Curve")
    ax.set_ylabel('Score')
    ax.set_xlabel('Training Examples')
    ax.legend(loc=1, fontsize='medium')
    
def classification_stats(y_val, model_pred, model_probs_target):
    """Calculate and present some useful classification statistics.
    It takes the true y values (y_val), the predicted y values (model pred),  
    the predicted probability of y=1 for the model (model_probs_target). 
    It returns a list of the classification accuracy, the true negative rate, 
    the true positive rate, the AUC, and the confusion matrix."""
    
    classification_accuracy = accuracy_score(y_val, model_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, model_pred).ravel()
    true_neg_rate = tn / (tn + fp)
    true_pos_rate = tp / (tp + fn)
    auc = roc_auc_score(y_val, model_probs_target[:, 1])
    con_matrix = confusion_matrix(y_val, model_pred)
    
    return [classification_accuracy, true_neg_rate, true_pos_rate, auc, con_matrix]
```

## Train-Validation-Test Split
I split the Varol et al. (2017) data into two parts. The training and validation set is 70% of the data. I use k-fold cross-validation to choose the parameters for a _particular_ model (e.g., choosing the number of features in a logistic regression model). After the parameters of the model have been chosen then I estimate the out-of-sample performance using the test set which is the remaining 30% of the data. This test set should give a fair reflection of the performance of the model on unseen data and I use this to choose _between_ models (e.g., random forests vs. logistic regression). 


```python
X_train_full, X_test, y_train_full, y_test = train_test_split(training_df[training_df.columns.difference(['bot'])], 
                                                              training_df['bot'], shuffle=True, 
                                                              test_size=0.3, random_state=4)
X_train_full = X_train_full.dropna(how='any')
y_train_full = y_train_full[X_train_full.index]
```

## Logistic Regression
To start with, I will fit a simple Logistic Regression model with all of the features and then perform some model diagnostics to try to see if we can improve performance.


```python
# Initialize cross-validation and logistic regression
cv = StratifiedKFold(n_splits=5)
logistic = LogisticRegression(random_state=4)

data = {}

# Perform cross-validation
f, (ax1, ax2) = plt.subplots(1, 2)
fold_num = 1
for temp_train_index, temp_val_index in cv.split(X_train_full, y_train_full):
    logistic.fit(X_train_full.iloc[temp_train_index], y_train_full.iloc[temp_train_index])
    logistic_pred_val = logistic.predict(X_train_full.iloc[temp_val_index])
    logistic_probs_val = logistic.predict_proba(X_train_full.iloc[temp_val_index])
    data[fold_num] = classification_stats(y_train_full.iloc[temp_val_index], logistic_pred_val, logistic_probs_val)
    draw_roc_curve(y_train_full.iloc[temp_val_index], logistic_probs_val[:, 1], ax1, fold_num)
    fold_num += 1

draw_learning_curve(logistic, X_train_full, y_train_full, ax2)
print('Average Cross Validation Score {:.4f}'.format(cross_val_score(logistic, 
                                                     X_train_full, y_train_full, cv=5).mean()))
pd.DataFrame(list(data.values()), 
             columns=['classification_accuracy', 'true_neg_rate', 
                      'true_pos_rate', 'AUC', 'confusion_matrix'],
            index=list(data.keys()))
```

Average Cross Validation Score 0.7246

<img src="/assets/trump_fakeaccounts_part2_files/figure6.png"/>

![png](/assets/trump_fakeaccounts_part2_files/trump_fakeaccounts_part2_21_2.png)


### Model Diagnostics

The learning curve shows that there is only a very small gap between the training score and the cross-validation score and so there doesn't appear to be a high variance problem. This suggests that gathering more data would not be helpful. The training and cross-validation scores also appear to be quite high (i.e., the errors are low) so there does not appear to be a bias problem. Therefore, it's not clear what we can do to improve this model so let's move on to the Random Forest model.

## Random Forest


```python
cv = StratifiedKFold(n_splits=5)
rf = RandomForestClassifier(random_state=4)

data = {}

# Perform cross-validation
f, (ax1, ax2) = plt.subplots(1, 2)
fold_num = 1
for temp_train_index, temp_val_index in cv.split(X_train_full, y_train_full):
    rf.fit(X_train_full.iloc[temp_train_index], y_train_full.iloc[temp_train_index])
    rf_pred_val = rf.predict(X_train_full.iloc[temp_val_index])
    rf_probs_val = rf.predict_proba(X_train_full.iloc[temp_val_index])
    data[fold_num] = classification_stats(y_train_full.iloc[temp_val_index], rf_pred_val, rf_probs_val)
    draw_roc_curve(y_train_full.iloc[temp_val_index], rf_probs_val[:, 1], ax1, fold_num)
    fold_num += 1

draw_learning_curve(rf, X_train_full, y_train_full, ax2)
print('Average Cross Validation Score {:.4f}'.format(cross_val_score(rf, 
                                                     X_train_full, y_train_full, cv=5).mean()))
pd.DataFrame(list(data.values()), 
             columns=['classification_accuracy', 'true_neg_rate', 
                      'true_pos_rate', 'AUC', 'confusion_matrix'],
             index=list(data.keys()))
```

Average Cross Validation Score 0.7517

<img src="/assets/trump_fakeaccounts_part2_files/figure7.png"/>

![png](/assets/trump_fakeaccounts_part2_files/trump_fakeaccounts_part2_24_2.png)


### Model Diagnostics
The random forests classifier achieves an almost perfect score on the training dataset but the score is much lower on the cross-validation set. This suggests that the model is overfitting the training data and that we have a high variance problem rather than a high bias problem. It may be possible to solve this problem by gathering more data or by reducing the complexity of the model. I can't gather more training data in this case so I will try to reduce the model complexity by reducing the number of features. In a Random Forests model we can get information on the importance of each of the features. I will use the top 5 most important features in the new Random Forests model. 


```python
# Which features are most important in the random forests model above?
feature_ranking = sorted(list(zip(X_train_full.columns, rf.feature_importances_)), 
                         key=lambda tup: tup[1], reverse=True)
for feature, importance in feature_ranking[:5]:
    print("{}: {:.4f}".format(feature, importance))

# Keep top 5 features
top5_features = [tup[0] for tup in feature_ranking[:5]]

X_train_rf_top5 = X_train_full.loc[:, top5_features].dropna(how='any')
y_train_rf_top5 = y_train_full.loc[X_train_rf_top5.index]
```
<img src="/assets/trump_fakeaccounts_part2_files/figure8.png"/>

```python
data = {}

# Perform cross-validation
f, (ax1, ax2) = plt.subplots(1, 2)
fold_num = 1
for temp_train_index, temp_val_index in cv.split(X_train_rf_top5, y_train_rf_top5):
    rf.fit(X_train_rf_top5.iloc[temp_train_index], y_train_rf_top5.iloc[temp_train_index])
    rf_pred_val = rf.predict(X_train_rf_top5.iloc[temp_val_index])
    rf_probs_val = rf.predict_proba(X_train_rf_top5.iloc[temp_val_index])
    data[fold_num] = classification_stats(y_train_rf_top5.iloc[temp_val_index], 
                                          rf_pred_val, rf_probs_val)
    draw_roc_curve(y_train_rf_top5.iloc[temp_val_index], rf_probs_val[:, 1], ax1, fold_num)
    fold_num += 1

draw_learning_curve(rf, X_train_rf_top5, y_train_rf_top5, ax2)
print('Average Cross Validation Score {:.4f}'.format(cross_val_score(rf, 
                                                     X_train_rf_top5, y_train_rf_top5, cv=5).mean()))
pd.DataFrame(list(data.values()), 
             columns=['classification_accuracy', 'true_neg_rate', 
                      'true_pos_rate', 'AUC', 'confusion_matrix'],
            index=list(data.keys()))
```

Average Cross Validation Score 0.7443

<img src="/assets/trump_fakeaccounts_part2_files/figure9.png"/>

![png](/assets/trump_fakeaccounts_part2_files/trump_fakeaccounts_part2_27_2.png)


There does not appear to be any improvement by reducing the complexity of the model (and I get similar results using only the top 3 or top 10 features). Ideally, we could get more training data to try to solve this problem but this is not possible in this case.

# Choosing the Model
So far we have trained our two models and tried to diagnose and improve their performance. Now let's see how each of the models perform on the test dataset which was not used in training or diagnosing the models. We achieved the best performance for both the Logistic Regression and the Random Forests when using all of the features so we will first train the models using the full training dataset (i.e., without any cross-validation this time):


```python
logistic = LogisticRegression(random_state=4)
logistic.fit(X_train_full, y_train_full)

rf = RandomForestClassifier(random_state=4)
rf.fit(X_train_full, y_train_full);
```

Now let's test the models using the test dataset and compare performance:


```python
X_test = X_test.dropna(how='any')
y_test = y_test[X_test.index]
```


```python
logistic_pred_test = logistic.predict(X_test)
logistic_probs_test = logistic.predict_proba(X_test)

accuracy_score(y_test, logistic_pred_test) # Same as (y_test == logistic_pred_test).mean()
```

0.75428571428571434


```python
rf_pred_test = rf.predict(X_test)
rf_probs_test = rf.predict_proba(X_test)

accuracy_score(y_test, rf_pred_test) # Same as (y_test == rf_pred_test).mean()
```

0.74285714285714288


We achieve remarkably good performance using both the logistic regression and random forests model on the test data. The performance of both models is very close to the cross-validated scores on the training dataset. The logistic regression has a slightly higher accuracy score on the test set so we will use this model to estimate the number of fake followers.

# Trump Followers Data
Now we can apply the model that we have trained to the dataset that we collected in the previous post. First, we need to apply the same preprocessing and feature engineering methods that we used for the training dataset:


```python
trump_followers_df = pd.read_csv('Data/trump_followers/TrumpFollowers_cleaned.csv', parse_dates=['created_at'])

# Perform same preprocessing as above
trump_followers_df['account_age_days'] = (datetime.datetime(2017, 12, 18) - trump_followers_df['created_at']).dt.days
trump_followers_df = trump_followers_df.drop(labels=['created_at'], axis=1)
trump_followers_df['followers_friends_ratio'] = ((trump_followers_df['followers_count'] / trump_followers_df['friends_count'])
                                          .replace(np.inf, np.nan)) # remove any infinity values
trump_followers_df['description_len'] = trump_followers_df['description'].str.len().replace(np.nan, 0) 
trump_followers_df = trump_followers_df.drop(labels=['id', 'name', 'screen_name', 'description', 'lang',
                                     'location', 'time_zone'], axis=1)

trump_followers_df = trump_followers_df.dropna(how='any')
```

```python
rf_pred_trump = rf.predict(trump_followers_df)
logistic_pred_trump = logistic.predict(trump_followers_df)

logistic_pred_trump.mean()
```

0.37424151781898984


So using the logistic regression model we estimate that between 37.42% of Trump's followers might not be real accounts. This estimates are broadly in line with previous work. However, we should note several limitations of this analysis. 

First, the nature of fake accounts on Twitter is constantly evolving and the techniques used to escape detection are increasingly sophisticated. This means that using the dataset compiled by Varol et al. (2017) may cause us to miss out on some of the newer bots. Additionally, the behaviour of bots captured in the Varol et al. data may differ from the bots that follow Trump.

Second, we used to Twitter API to get a sample of the 40 million accounts that follow Trump but, as noted in the previous blog post, the sample was not random. Instead, the ``Cursor`` object returns followers in the order in which they were added. This could bias our estimate of the number of Trump fake followers. For example, if Trump's more recent followers are more likely to be bots then we may overestimate the number of fake accounts that follow Trump. 

Third, the sample size was also very small (approx. 11,000). It seems likely that the estimate of Trump's fake following could change significantly if we gathered another sample. To get a more stable estimate of the number of fake accounts following Trump we may need a much larger sample.

