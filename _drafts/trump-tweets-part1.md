---
layout: post
author_profile: true
title: ''
description: 'Description of test post'
output: html_document
comments: true
---


There have been several claims that many of Donald Trump's Twitter followers are fake. For example, the <a href="https://www.twitteraudit.com/realdonaldtrump"  style="color: rgb(0,0,0)">twitteraudit</a> service calculates that 20.8 million of Trump's 43.8 million followers are fake. Given that Twitter's API is quite friendly to use, I thought it would be fun to try to collect some data from Twitter and use some simple machine learning techniques to see if we can come up with our own estimate of how many fake followers Trump has.

In this first post I will collect the data from Twitter using it's API. The next post will apply the machine learning models.

I will use the <a href="https://www.tweepy.org"  style="color: rgb(0,0,0)">tweepy</a> module to connect to the Twitter API. The first thing we need to do is to <a href="https://apps.twitter.com/" style="color: rgb(0,0,0)">register</a> an app with Twitter to be able to connect to the Twitter API. When you do this you should be able to access your consumer and secret keys which you should keep private. The code below then allows you to gain authorized  access to Twitter's API. I have replaced my access codes with empty strings and you should insert your codes in here.


```python
# import some modules that we will use
import datetime
import time
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from tweepy import RateLimitError
import csv
import pandas as pd
import os.path

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

The next thing we need is a list of Trump's followers. To keep the analysis quick and simple for now I won't try to collect a list of all of Trump's followers but I will instead take a sample and see what we can extrapolate from this. The code below defines a function ``get_followers`` which takes the user ID of any (public) Twitter account, collects certain information about ``num_followers`` from that account, and stores this information in a csv file:


```python
def get_followers(user_id, num_followers, output_file='followers.csv'):
    """Takes a user id from a Twitter account and puts information on 
    num_follower followers into a csv file."""
    
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
    # information into the csv file
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for follower in Cursor(myAPI.followers, id=user_id).items(num_followers):
            writer.writerow([follower.name, follower.id_str, follower.created_at,
                            follower.default_profile, 
                            follower.default_profile_image, 
                            follower.description, follower.favourites_count,
                            follower.followers_count, follower.friends_count,
                            follower.geo_enabled, follower.lang, 
                            follower.location, follower.protected, 
                            follower.screen_name, follower.statuses_count, 
                            follower.time_zone, follower.verified])
```

For example, ``get_followers(user_id=100, num_followers=10, output_file='followers.csv')`` collects information on 10 followers of the account with the user ID 100 and stores this information in the ``followers.csv`` file. 

Next we should have a look at the data we collected for Trump's Twitter account  to make sure there are no obvious errors that could cause problems for our models later. A convenient way to do this is to load the data into a [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) DataFrame.


```python
trump_followers_df = pd.read_csv('TrumpFollowers.csv')
trump_followers_df['created_at'] = pd.to_datetime(trump_followers_df['created_at'])

trump_followers_df.dtypes
```




    name                             object
    id                                int64
    created_at               datetime64[ns]
    default_profile                    bool
    default_profile_image              bool
    description                      object
    favourites_count                  int64
    followers_count                   int64
    friends_count                     int64
    geo_enabled                        bool
    lang                             object
    location                         object
    protected                          bool
    screen_name                      object
    statuses_count                    int64
    time_zone                        object
    verified                           bool
    dtype: object




```python
bool_cols = trump_followers_df.columns[trump_followers_df.dtypes == 'bool']
trump_followers_df.loc[:, bool_cols] = trump_followers_df.loc[:, bool_cols].astype(int)
```

Let's take a look at some summary statistics for some of the numeric columns.


```python
num_cols = ['default_profile', 'default_profile_image', 'favourites_count', 'followers_count', 
            'friends_count', 'geo_enabled', 'protected', 'statuses_count', 'verified']
trump_followers_df[num_cols].describe().round(2)
```

![My helpful screenshot]({{ "/assets/ss.png" | absolute_url }})
