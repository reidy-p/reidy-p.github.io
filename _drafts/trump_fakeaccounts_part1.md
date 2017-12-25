---
layout: post
author_profile: true
title: "Estimating Trump's Fake Twitter Followers: Part 1"
output: html_document
date: 2017-12-23 12:45:00 -0400
---

Donald Trump enjoys using his [Twitter account](https://twitter.com/realDonaldTrump/) to ["go around"](https://twitter.com/realDonaldTrump/status/875690204564258816) the "Fake News Media" and has amassed a large following on Twitter. But some [claim](https://www.dailydot.com/layer8/trump-fake-twitter-followers-october-2017/) that a large portion of Trump's Twitter followers are fake. At the time of writing, the [twitteraudit](https://www.twitteraudit.com/realdonaldtrump) service, for example, calculates that 20.8 million of Trump's 43.8 million followers are fake. 

I thought it would be fun to try to collect some data from Twitter and use some simple machine learning techniques to see if we can come up with our own estimate of how many fake followers Trump has. This is particularly attractive because Twitter's API is quite friendly to use. In this first post I will collect the data from Twitter using it's API and the next post will apply the machine learning models to try to estimate the extent of Trump's fake followers.

I will use the [tweepy](https://www.tweepy.org) module to connect to the Twitter API. The first thing we need to do is to [register](https://apps.twitter.com/) an app with Twitter to be able to connect to the Twitter API. When you do this you should be able to access your consumer and secret keys which you should keep private. The code below then allows you to gain authorized  access to Twitter's API. I have replaced my access codes with empty strings and you should insert your codes in here.


```python
# import some modules that we will use
import datetime
import time
from tweepy import OAuthHandler, API, Cursor, RateLimitError
import csv
import pandas as pd
import numpy as np
import os.path
```


```python
# put your information into these empty strings
consumer_key = ''
consumer_secret = '' 
access_token = ''
access_token_secret = ''

# gain authorized access to the Twitter API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
myAPI = API(auth)

# get the account ID for Donald Trump's Twitter account
trump_account_id = myAPI.get_user('realDonaldTrump').id
```

## Trump's Followers Data
The next thing we need is a list of Trump's followers. To keep the analysis quick and simple I won't try to collect a list of all of Trump's 40 million followers but I will instead take a sample and see what we can extrapolate from this. The code below defines some functions which takes the user ID of any (public) Twitter account, collects certain information about ``num_followers`` from that account, and stores this information in a csv file:


```python
def limit_handled(cursor):
    """A convenient way to handle rate limits with the tweepy Cursor.
    Taken from the tweepy docs."""
    while True:
        try:
            yield cursor.next()
        except RateLimitError:
            print('Taking a 15 minute break to avoid rate limit')
            time.sleep(15 * 60)

def get_account_info(account, api, english_only=True):
    """Get information on a Twitter account. This function accepts
    a tweepy User object or an account ID."""

    # Convert an account ID into a user object if necessary
    if not isinstance(account, tweepy.models.User):
        account = api.get_user(account)

    if account.lang.startswith('en'):
        return [account.name, account.id_str, account.created_at,
                account.default_profile, account.default_profile_image,
                account.description, account.favourites_count,
                account.followers_count, account.friends_count,
                account.geo_enabled, account.lang,
                account.location, account.protected,
                account.screen_name, account.statuses_count,
                account.time_zone, account.verified]

def output_followers(user_id, num_followers, api, output_file='followers.csv'):
    
    # Create csv file if it doesn't exist
    if not os.path.exists(output_file):    
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            # Headings for the csv file
            writer.writerow(['name', 'id', 'created_at', 'default_profile',
                            'default_profile_image', 'description', 
                            'favourites_count', 'followers_count', 
                            'friends_count', 'geo_enabled', 'lang', 'location', 
                            'protected', 'screen_name', 'statuses_count', 
                            'time_zone', 'verified'])
    
    # Iterate through the number of followers specified and write follower 
    # information to the csv file
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        cursor_obj = Cursor(api.followers, id=user_id).items(num_followers)
        for count, follower in enumerate(limit_handled(cursor_obj)):
            count += 1
            print(f'Examining follower {count} out of {num_followers}...')
            if follower.lang.startswith('en'):
                writer.writerow(get_account_info(follower, api))
```

For example, ``output_followers(user_id=100, num_followers=10, output_file='followers.csv')`` collects information on 10 followers of the account with the user ID 100 and stores this information in the ``followers.csv`` file. One of the benefits of using the ``tweepy`` library is that it provides a ``Cursor`` object which allows us to iterate quite easily through the desired number of followers. 

Ideally, we would like to get a random sample of Trump's followers. This would allow us to use a small, representative sample and try to extrapolate about all of Trump's followers. However, when we call the ``followers`` method on an instance of the ``API`` class from the ``tweepy`` module, it returns the user's followers in the order in which they were added. This means that we should be careful about reading too much into our results - I will return to this topic at the end of the next post.

Next we should have a look at the data we collected for Trump's Twitter account  to make sure there are no obvious errors that could cause problems for our models later. A convenient way to do this is to load the data into a [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) DataFrame:


```python
trump_followers_df = pd.read_csv('Data/trump_followers/TrumpFollowers_raw.csv')

# Convert created_at to datetime
trump_followers_df['created_at'] = pd.to_datetime(trump_followers_df['created_at'])

# Remove duplicated rows
trump_followers_df = trump_followers_df.drop_duplicates(subset='id', keep='first')

# Check that all the dtypes make sense
trump_followers_df.dtypes
```
<img src="/assets/trump_fakeaccounts_part1_files/figure1.png"/>

Now let's take a look at some summary statistics for some of the numeric columns.


```python
# Convert boolean columns to int
bool_cols = trump_followers_df.select_dtypes(include=[bool]).columns
trump_followers_df.loc[:, bool_cols] = trump_followers_df.loc[:, bool_cols].astype(int)

num_cols = ['default_profile', 'default_profile_image', 'favourites_count', 'followers_count', 
            'friends_count', 'geo_enabled', 'protected', 'statuses_count', 'verified']
trump_followers_df[num_cols].describe().round(2)
```

<img src="/assets/trump_fakeaccounts_part1_files/figure2.png"/>


We can see that we have collected data on 11,702 accounts, excluding the duplicated rows. The summary statistics all seem to be in line with what we might expect. For example, the accounts have posted 175.96 tweets, have 110.12 friends (other accounts that a particular account follows), and 54.01 followers, on average. Interestingly, we can see that 60% of the accounts in the sample have the default profile picture which seems very high and may provide an initial clue that there are many fake accounts in the dataset.


```python
trump_followers_df.to_csv('Data/trump_followers/TrumpFollowers_cleaned.csv', index=False)
```

## Training Data
We now have a dataset of accounts that follow Trump on Twitter. But to apply a supervised learning model to this problem we need to get a dataset that is already labelled. That is, we need a dataset where the accounts have already been classified as real or fake. We can then use this data to help the models "learn" how to distinguish between the real and fake accounts in the Trump dataset.

We have several options for getting a training dataset. In some academic papers, for example, a portion of the dataset is classified manually and is used to train the models. [Erin Shellman](http://www.erinshellman.com/bot-or-not/), on the other hand, uses the website [fiverr](https://www.fiverr.com/) to purchase fake followers. We could also use services like [botometer](https://botometer.iuni.iu.edu/#!/) to try to estimate how likely a particular account is to be fake.

I will use the labelled [dataset](https://botometer.iuni.iu.edu/bot-repository/datasets.html) from a paper by [Varol et al. (2017)](https://arxiv.org/pdf/1703.03107.pdf). Let's take a look at the dataset:


```python
varol_original_df = pd.read_excel('Data/varol2017_data/varol2017_original_data.xlsx', 
                                  header=None, names=['account_id', 'bot'])
varol_original_df.head()
```

<p style="text-align:left;">
<img src="/assets/trump_fakeaccounts_part1_files/figure3.png" width="200" height="100"/>
</p>

We can see that the dataset consists of an ``account_id`` variable and a ``bot`` variable where a value of 1 means that the account has been labelled as a fake account and 0 means that it has been labelled as real. However, to use this as a training dataset for classifying the Trump followers we need to get the same features that we collected above so that the models can learn how the different features can distinguish between real and fake accounts. The code below allows us to gather the same features from all of the accounts listed in the Varol et al. dataset.


```python
varol_original_df.index = varol_original_df.index + 1 # make the index start from 1

# The script is very time consuming so it will probably take multiple runs of
# the script. Here I check if the file has already been started and how much
# progress made. I then use this to continue where I finished previously.
row_count = 0
if os.path.exists('Data/varol2017_data/varol2017_training_data.csv'):
    with open('Data/varol2017_data/varol2017_training_data.csv', 'r') as f:
        row_count = sum(1 for row in f) - 1 # Subtract 1 to avoid counting heading
    print(f"Previously stopped at account {row_count} out of {varol_original_df.shape[0]}")

# Create column names if file doesn't already exist
else:
    with open('Data/varol2017_data/varol2017_training_data.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['bot', 'name', 'account_id', 'created_at', 
                        'default_profile', 'default_profile_image', 'description', 
                        'favourites_count', 'followers_count', 'friends_count', 
                        'geo_enabled', 'lang', 'location', 'protected', 
                        'screen_name', 'statuses_count', 'time_zone', 
                        'verified'])

remaining_varol_original_df = varol_original_df[row_count:]

with open('Data/varol2017_data/varol2017_training_data.csv', 'a', newline='') as f:
    writer = csv.writer(f, delimiter=',')

    # Iterating over the rows of the dataframe. Could this be vectorised?
    for row in remaining_varol_original_df.itertuples():

        print(f"Gathering features for account {row.Index} "
              f"out of {varol_original_df.shape[0]}...")

        try:
            if myAPI.get_user(row.account_id).lang.startswith('en'):
                writer.writerow([row.bot] + 
                                get_account_info(row.account_id, myAPI))

        except RateLimitError as e:
            print('Rate limit exceeded. Waiting for 15 minutes...')
            time.sleep(15 * 60)
            if myAPI.get_user(row.account_id).lang.startswith('en'):
                writer.writerow([row.bot] + 
                                get_account_info(row.account_id, myAPI))

        except TweepError as e:
            print("Failed!", e.response.text)
            print()
```

Now let's take a look at the modified Varol et al. dataset that we are going to use:


```python
varol_training_df = pd.read_csv('Data/varol2017_data/varol2017_training_data.csv')
varol_training_df['created_at'] = pd.to_datetime(varol_training_df['created_at'])

varol_training_df.dtypes
```

<img src="/assets/trump_fakeaccounts_part1_files/figure4.png"/>


```python
# Convert boolean columns to int (0 or 1)
bool_cols = varol_training_df.columns[varol_training_df.dtypes == 'bool']
varol_training_df.loc[:, bool_cols] = varol_training_df.loc[:, bool_cols].astype(int)

print("There are {} accounts labelled human and {} labelled as "
      "a bot in the Varol et al. (2017) training data.".format(*varol_training_df['bot'].value_counts()))

num_cols = ['default_profile', 'default_profile_image', 'favourites_count', 'followers_count', 
            'friends_count', 'geo_enabled', 'protected', 'statuses_count', 'verified']
varol_training_df[num_cols].describe().round(2)
```

There are 1632 accounts labelled human and 736 labelled as a bot in the Varol et al. (2017) training data.

<img src="/assets/trump_fakeaccounts_part1_files/figure5.png"/>


There were 2573 accounts in the original Varol et al. dataset but some of those accounts no longer exist (i.e., they were deleted or suspended) or do not tweet in English so the modified dataset has 2368 items. 

Next let's look at some summary statistics broken down by whether the account is labelled as real or fake:


```python
num_cols = ['bot', 'default_profile', 'default_profile_image', 'favourites_count', 'followers_count', 
            'friends_count', 'geo_enabled', 'protected', 'statuses_count', 'verified']

(varol_training_df[num_cols].groupby('bot')
                            .describe()
                            .loc[:, (slice(None), ['mean', 'std', 'max'])]
                            .loc[:, (slice('default_profile', 'verified'), ['mean', 'std', 'max'])]
                            .round(2)
                            .T)
```

<img src="/assets/trump_fakeaccounts_part1_files/figure6.png"/>


As we might expect, the accounts labelled as fake (bot = 1) are more likely to have the default profile settings, the default profile picture, are less likely to be protected, and have more followers and friends, on average. It may surprise us that the fake accounts post fewer statuses, on average, because we usually associate such accounts with generating a lot of spam.

We can also visualise these differences to see if we can get a clearer picture.


```python
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
plt.style.use('fivethirtyeight')
%matplotlib inline
rcParams['figure.figsize'] = (10, 10)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
sns.barplot(x=['bot = 0', 'bot = 1'], y=varol_training_df.groupby('bot')['default_profile'].mean()*100, ax=ax1)
ax1.set_title('Percentage of Profiles with Default Settings')
ax1.set_ylabel('Percent (%)')
sns.barplot(x=['bot = 0', 'bot = 1'], y=varol_training_df.groupby('bot')['default_profile_image'].mean()*100, ax=ax2)
ax2.set_title('Percentage of Profiles with Default Image')
ax2.set_ylabel('Percent (%)')
sns.barplot(x=['bot = 0', 'bot = 1'], y=varol_training_df.groupby('bot')['verified'].mean()*100, ax=ax3)
ax3.set_title('Percentage of Profiles Verified')
ax3.set_ylabel('Percent (%)')
sns.barplot(x=['bot = 0', 'bot = 1'], y=varol_training_df.groupby('bot')['protected'].mean()*100, ax=ax4)
ax4.set_title('Percentage of Profiles Protected')
ax4.set_ylabel('Percent (%)');
```


![png](/assets/trump_fakeaccounts_part1_files/trump_fakeaccounts_part1_21_0.png)



```python
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    
# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
formatter = FuncFormatter(to_percent)
    
# Make a normed histogram. It'll be multiplied by 100 later.
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.hist(varol_training_df.loc[varol_training_df['bot'] == 0,'friends_count'].dropna(), 
         range=(0, 6000), color='blue', bins=50, normed=True)
ax1.set_title('No. of Friends (Bot = 0)')
ax1.yaxis.set_major_formatter(formatter)
                                                                                             
ax2.hist(varol_training_df.loc[varol_training_df['bot'] == 1,'friends_count'].dropna(), 
         range=(0, 6000), color='red', bins=50, normed=True)
ax2.set_title('No. of Friends (Bot = 1)')
ax2.yaxis.set_major_formatter(formatter)
                                                                                             
ax3.hist(varol_training_df.loc[varol_training_df['bot'] == 0,'followers_count'].dropna(), 
         range=(0, 6000), color='blue', bins=50, normed=True)
ax3.set_title('No. of Followers (Bot = 0)')
ax3.yaxis.set_major_formatter(formatter)

ax4.hist(varol_training_df.loc[varol_training_df['bot'] == 1,'followers_count'].dropna(), 
         range=(0, 6000), color='red', bins=50, normed=True)
ax4.set_title('No. of Followers (Bot = 1)')
ax4.yaxis.set_major_formatter(formatter)
                                                                                             
ax5.hist(varol_training_df.loc[varol_training_df['bot'] == 0, 'statuses_count'].dropna(), color='blue', 
         range=(0, 30000), bins=50, normed=True)
ax5.set_title('No. of Tweets (Bot = 0)')
ax5.yaxis.set_major_formatter(formatter)
                                                                                             
ax6.hist(varol_training_df.loc[varol_training_df['bot'] == 1, 'statuses_count'].dropna(), color='red', 
         range=(0, 30000), bins=50, normed=True)
ax6.set_title('No. of Tweets (Bot = 1)')
ax6.yaxis.set_major_formatter(formatter);
```


![png](/assets/trump_fakeaccounts_part1_files/trump_fakeaccounts_part1_22_0.png)


The histograms don't reveal any clear differences for the distributions of number of friends, followers, and tweets.

# Conclusion
In this post, I collected data on a small sample of Donald Trump's Twitter followers and also prepared a training dataset. In the next post I will use the training dataset to train and test several models and then classify the Trump followers that we sampled as real or fake.
