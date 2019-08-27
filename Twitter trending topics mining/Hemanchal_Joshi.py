import pandas as pd
import tweepy
import csv
import json
from newsapi.newsapi_client import NewsApiClient

# setting up token keys
consumer_key = 'bnd2051rr93p6o15qssGU3uTs'
consumer_secret = 'dwmvd2ql39lIr1lzXqdt4pDWv3jLpvYILKV0teHf21d2jDBOi8'
key = '720117643-FBi1huP7z3pFxQb2aD6WAe6nBSHQM8uV8mzMlWOz'
secret = '1Vx01tP6RxfEiNlgF10Qenznu8YXZAgp8jiJpCqx7wOZ9'
news = NewsApiClient(api_key='f8f7b62b105f495a977efddb4a0222f5')

# Authentication to twitter API
authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(key, secret)
api = tweepy.API(authentication)

# Fetch the trending topics in twitter
res = api.trends_place(1, exlude='#')

# Define the tuples
records = res[0]['trends']

# Fetch the name for each trend and print it
names = [record['name'] for record in records]

# data = json.dumps(res, indent=1)
# Looping on each 10 trending topics in twitter
for i in range(0,10):
    topic = names[i]
    print(topic)

    # Fetching the related articles from news api on the basis of the twitter name topic
    top_headlines = news.get_top_headlines(q=topic, language='en')

    # Count the total number of articles available
    count = top_headlines['totalResults']

    # CHeck if there are any related articles available
    if count != 0:
        # Print in console and Write the available output in the output file
        with open('output.txt','a') as f:
            print(topic)
            print(json.dumps(top_headlines, indent=1))
            print("-----------------------------------")
            print("-----------------------------------")
