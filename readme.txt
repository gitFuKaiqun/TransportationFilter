transportation_tweet_filter.py is the code that can scan U.S. twitter data files in an InputFolder, and then output transportation related tweets to an OutPutFolder. 

folder "filtered data" stores some example transportation realted tweet files organized in day by day files. 
use json load, and it is a list of raw tweet files [tweet_1, tweet_2, ...], tweet_i is a dictionary format of tweet, and you can read the attributes, such as tweet_1['user']['location']


