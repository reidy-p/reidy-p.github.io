---
layout: post
title: Strava Stats Flask App
date: 2019-10-05
---
In this post I will highlight some of the interesting features and code involved in creating the Strava Stats. This app is still under development but if you have any feedback I'd love to hear it on [GitHub](https://github.com/reidy-p/strava-stats/issues) or by <a href="mailto:paul_reidy@outlook.com">e-mail</a>.

DarkSky API
---


VDOT Calculation
---
[VDOT]("https://en.wikipedia.org/wiki/Jack_Daniels_(coach)#VDOT") is an estimate of overall fitness and is used by many runners to calculate equivalent performances and recommended training paces. For example, if you recently ran a 5km race in 25 minutes you can calculate your VDOT as 38.3 using [this VDOT calculator](https://runsmartproject.com/calculator/) and find training paces and equivalent race performances for other distances for the same VDOT value. In the app I was developing I wanted to include a feature to automatically calculate the VDOT based on all activities that were uploaed to Strava and marked as races. This would allow you to estimate overall fitness at different points in time. One method of doing this would be to manually create VDOT tables. A second option was to embed the [runsmartproject](https://runsmartproject.com/calculator/) calculator into the app. My preferred option was an external API that I could call but I couldn't find any. I did notice that using the runsmartproject calculator made a POST request to the  'https://runsmartproject.com/vdot/app/api/find_paces' so I managed to replicate these POST requests in Python using the code below. This meant that I could use this calculator like an API instead of having to embed it into the web app.

```python
def calculate_vdot(distance_metres, moving_time):
    """
    Calculate VDOT using the runsmartproject calculator. There is no public
    API for this calculator so I reverse engineered how the POST request
    seems to work to get the VDOT calculations and equivalent race results 
    """

    request_data = {
      'distance': round(distance_metres, -2) / 1000,
      'unit': 'km',
      'time': moving_time
    }
    
    vdot_data = requests.post('https://runsmartproject.com/vdot/app/api/find_paces', 
                              data=request_data)

    return (vdot_data.json()['vdot'], vdot_data.json()['paces']['equivs'])
```

The VDOT page in the Strava Stats is shown in the screenshot below:

![jpg](/static/VDOT.jpg)

Progress Bar for Downloads
---
The Strava Stats app downloads data on each of the activities that you have uploaded to Strava and stores them locally in a SQLite database. This downloading can take a while if you have uploaded hundreds of activities. In the first version of the app the browser would simply wait for the download to complete with no indication of the progress so I decided that a better design would show a progress bar for the downloading. This proved to be more difficult to implement than expected. After doing some research I found that the best way of implementing this involved the following steps:
* In one thread start the downloading of the activity data
* Setup a Redis database and store a value for the progress of the download (percentage of downloads complete) in that database 
* In a separate thread query the Redis database at regular intervals to get the progress percentage and show a progress bar based on this value

