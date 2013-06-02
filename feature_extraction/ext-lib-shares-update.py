# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:06:32 2012

@author: Feng Chen
"""
import nltk

#nltk.download()

from numpy.random import *
from numpy import *
from scipy import spatial
from datetime import *
from dateutil import parser
from itertools import groupby
from operator import itemgetter
from scipy import *
import csv
import re
import os
import csv
#import unicodecsv
import utils
import sys
import glob
import geocode
import shutil
import nltk
import math
from collections import Counter
from math import sin, cos, radians, atan2
from utils import normalize_payload, normalize_str
import json
import copy
from urlparse import urlparse
import re
import pickle
import cPickle
from geocode import Geo
from scipy.spatial import KDTree
from nltk import SnowballStemmer
#from feature_extraction.text import FeatureCountVectorizer
import time
import warnings
import threading
import multiprocessing
import codecs
import sys
import utilscivil
from operator import itemgetter
from scipy.stats import norm
from Share_threshold import *

reload(sys)
sys.setdefaultencoding("utf-8")

#CITY_DATA = os.path.join(os.path.dirname(__file__), "data\city_data.txt")
#WG_DATA = os.path.join(os.path.dirname(__file__), "data\wg-partial.txt")

root = os.path.dirname(__file__)
latlong_2_city = pickle.load(open(root + '/data/latlong2citycountry.pkl'))


class Task_calc(object):
    def __init__(self, co_name, co_user_fld_name):
        self.co_name = co_name
        self.co_user_fld_name = co_user_fld_name
    def __call__(self):
        print self.co_name
        sys.stdout.flush()
        init_loc_users(self.co_name, self.co_user_fld_name)

    def __str__(self):
        return '%s processed!' % (self.co_name)

"""

Step 01: Generate "glb.glb_dict_countries"
Step 02: Generate "glb.glb_dict_latlongs"
Step 03: Generate "glb.glb_dict_users". Process all tweets and genereate the collections of users.
Step 04: Generate state neighborhood graph, and city neighborhood graph

Generate "glb.glb_dict_tweets", "glb.glb_dict_keywords", "glb.glb_dict_city_keywords"

Step 1: Collect the tweets for one whole day
Step 2: Parse all tweets and generate "glb.glb_dict_tweets", "glb.glb_dict_keywords", "glb.glb_dict_city_keywords"
Step 3: For each state or country, update glb.glb_dict_keywords
Step 4: Build cooccur_graph for each city, state, and country

"""


def init_loc_users(co_name, co_user_fld_name):

    output_folder  = os.path.join(root, 'stat', co_name)
    countries      = [co_name]
    in_folder      = os.path.join(root, 'data', co_name)
    if len(co_user_fld_name) > 0:
        follower_data_folder = os.path.join(root, 'data/usersLists/twitter_users_list/usersLists', co_user_fld_name)
    else:
        follower_data_folder = None
    print 'start initialization process'

    # Step 1: Initilize variables
    glb = glb_dict()

    # Step 2: Load dict_countries, dict_latlongs, dict_city_keywords
    glb.glb_dict_countries, glb.glb_dict_states, glb.glb_dict_cities = Share.load_dict_countries_states_cities(countries, glb)

    # Step 3: Load user ids from the raw tweets
    dict_user_ids = dict()
    dict_userid_2_scrname = dict()

    dict_sig_co_user_ids, dict_sig_userid_2_scrname = User.collect_user_ids(in_folder)
    dict_user_ids.update(dict_sig_co_user_ids)
    dict_userid_2_scrname.update(dict_sig_userid_2_scrname)

#    out_file = out_folder + '/dict_user_ids.pkl'
#    Share.save_to_pkl_file(dict_user_ids, out_file)
    glb.dict_userid_2_scrname = dict_userid_2_scrname
    
    print 'Step 4: Update follower relationships only for users in user_ids'
    # Step 4: Update follower relationships only for users in user_ids
    glb.arch_glb_dict_users = User.load_follow_relationships(follower_data_folder, dict_user_ids) # This step is to load the follower info. Note that, for now, each user only has an id and followers.
    
#    print len(glb.arch_glb_dict_users.items())
#    print len(glb.arch_glb_dict_users.keys())

    glb.glb_geo = None  # do not store kd tree in disk
    Share.new_folder(output_folder)
    Share.save_to_pkl_file(glb, os.path.join(output_folder, 'glb.pkl'))
    
    return glb
    
def cdf_calc(cdf_values, x):
    x = int(floor(x * 100))
    if x >= 400:
        return 1
    else:
        return cdf_values[x]

def multicore_proc(num_consumers):
    
    tasks   = multiprocessing.Queue()
    results = multiprocessing.Queue()
    
    # Start consumers
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    
    countries = ['argentina', 'costa rica', 'mexico', 'brazil', 'chile', 'colombia', 'el salvador', 'ecuador', 'paraguay', 'venezuela']
    co_user_fld_names = ['usersArgentina', 'usersCostaRica', 'usersMexico', '', '', 'usersColombia', 'usersElSalvador', 'usersEcuador', '', 'usersVenezuela']
    countries = ['argentina', 'mexico', 'brazil', 'chile', 'colombia', 'el salvador', 'ecuador', 'paraguay', 'venezuela']
    co_user_fld_names = ['usersArgentina', 'usersMexico', '', '', 'usersColombia', 'usersElSalvador', 'usersEcuador', '', 'usersVenezuela']
    countries = ['mexico']
    co_user_fld_names = ['usersMexico']
    countries = ['argentina', 'mexico', 'colombia', 'ecuador', 'venezuela']
    co_user_fld_names = ['usersArgentina', 'usersMexico', 'usersColombia','usersEcuador', 'usersVenezuela']
    num_jobs = len(countries)
    for co_name, co_user_fld_name in zip(countries, co_user_fld_names):
        if len(co_user_fld_name) > 0:
            tasks.put(Task_calc(co_name, co_user_fld_name))
    
    # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)
    
    while num_jobs:
        results.get()
        print 'num jobs: ', num_jobs
        num_jobs -= 1
    
    
def calc_stats():

    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[2:]:
        print co_name
        log_folder = os.path.join('D:/embers/stat', co_name)
        logging = Log(log_folder + '/evaluations_maxdata_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        logging_sum = Log(log_folder + '/evaluations_summary_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    #    return 
        desps = ['# tweets', '# retweets', '# followers', '# followees', '# mentioned_by', 'E replied_by', 'diff graph depth', 'diff graph size']
        logging_sum.add_line('{}'.format(desps))
        for di in range(120):
            if di > len(files) - 1:
                continue
            max_vals = []    
    #        agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
            print files[di]
            dt = datetime.strptime(files[di], "%Y-%m-%d")
            logging.add_empty_lines(4)
            logging.add_line('************************************')
            logging.add_line(files[di])
            logging_sum.add_line(files[di])
            for i in range(8):
                logging.add_empty_lines(2)
                max_v = None
                max_idx = None
    #            items = []
                for key, user in glb.glb_dict_users.items():
                    user.calc_features(glb)
                    if max_v == None or max_v < user.features[i]:
                        max_v = user.features[i]
                        max_idx = key
                max_vals.append(max_v)
                logging.add_line('{}: {}'.format(desps[i], glb.glb_dict_users[max_idx].features[i]))
                count = 0
                for key, user in glb.glb_dict_users.items():
                    user.calc_features(glb)
                    if max_v == user.features[i]:
                        count += 1
    #                    user = glb.glb_dict_users[max_idx]                                 
                        logging.add_line(user.get_basic_info())
                        for kt, tweet_id in user.dict_tweet_ids.items():
                            tweet = glb.glb_dict_tweets[kt]
            #                items.append(user.calc_features(glb)[i])
                            logging.add_line(tweet.text)
                        logging.add_empty_lines(1)
                        if count > 5:
                            break
                        
    #        print dt.date()
    #        print glb.gsr
            if glb.gsr.has_key(dt.date()):
                logging_sum.add_line('{event}')
            else:
                logging_sum.add_line('{no event}')
                
            logging_sum.add_line('{}'.format(max_vals))
    #            logging.add_line('max value: {}'.format(max(items)))
    #            logging.add_line('{}'.format(items))
            for key, link in glb.glb_dict_links.items():
        #        print link.httpquier
                if link.http == 'http://t.co/bfHiXukR':
                    logging.add_line('matched')
                    logging.add_line('{}'.format(len(link.link_tweet_ids)))
#    print items


def test_user_stats():

    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[0:1]:
        print co_name
        log_folder = os.path.join('D:/embers/stat', co_name)
        logging = Log(log_folder + '/evaluations_maxdata_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
#        logging_sum = Log(log_folder + '/evaluations_summary_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    #    return 
        desps = ['# tweets', '# retweets', '# followers', '# followees', '# mentioned_by', 'E replied_by', 'diff graph depth', 'diff graph size']
        logging.add_line('{}'.format(desps))
        for di in range(30):
            if di > len(files) - 1:
                continue
    #        agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
            print files[di]
            dt = datetime.strptime(files[di], "%Y-%m-%d")
            logging.add_empty_lines(4)
            logging.add_line('************************************')
            logging.add_line(files[di])
            max_diff_graph_size = 0
            for i in range(8):
                items = []
                for key, user in glb.glb_dict_users.items():
                    user.calc_features(glb)
                    items.append(user.features[i])
                logging.add_empty_lines(2)
                cp_items = copy.deepcopy(items)
                cp_items.sort(reverse = True)
                logging.add_line('{}: {}, count ({}), mean(std): {} ({}), median(mad): {} ({})'.format(desps[i], str(cp_items[0:10]), len(items), mean(items), std(items), median(items), median([abs(val - median(items)) for val in items])))
                logging.add_items(items)
                if i == 7:
                    max_diff_graph_size = max(items)
            if glb.gsr.has_key(dt.date()):
                logging.add_line('{event}')
            else:
                logging.add_line('{no event}')
            
            # print tweets that are related to the max_diff_graph_size
            for key, user in glb.glb_dict_users.items():
                if user.features[7] == max_diff_graph_size:
                    tweets = user.get_diffiusion_tweets(glb)
                    for tweet in tweets:
                        logging.add_line(tweet.text)
            
#            for key, link in glb.glb_dict_links.items():
#        #        print link.http
#                if link.http == 'http://t.co/bfHiXukR':
#                    logging.add_line('matched')
#                    logging.add_line('{}'.format(len(link.link_tweet_ids)))

def test_tweet_stats():

    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[0:1]:
        print co_name
        log_folder = os.path.join('D:/embers/stat', co_name)
        logging = Log(log_folder + '/evaluations_maxdata_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
#        logging_sum = Log(log_folder + '/evaluations_summary_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    #    return 
        desps = ['replied_by graphs size', 'reply graph depth', 'retweet graph size', 'retweet graph depth']
        logging.add_line('{}'.format(desps))
        for di in range(150):
            if di > len(files) - 1:
                continue
    #        agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
            print files[di]
            dt = datetime.strptime(files[di], "%Y-%m-%d")
            logging.add_empty_lines(4)
            logging.add_line('************************************')
            logging.add_line(files[di])
            for i in range(4):
#                logging.add_line(desps[i])
                items = []
                for key, tweet in glb.glb_dict_tweets.items():
                    tweet.calc_features(glb)
                    items.append((key, tweet.features[i]))
                logging.add_empty_lines(2)
                cp_items = copy.deepcopy(items)
                cp_items.sort(key = lambda x: x[1], reverse = True)
                values = [val[1] for val in cp_items]
                
                logging.add_line('{}: {}, count ({}), mean(std): {} ({}), median(mad): {} ({})'.format(desps[i], values[0:10], len(items), mean(values), std(values), median(values), median([abs(val - median(values)) for val in values])))

                if i == 2:
                    for j in range(10):
                        if j >= len(cp_items):
                            continue
                        key = cp_items[j][0]
                        tweet = glb.glb_dict_tweets[key]
                        if len(tweet.text) > 3:
                            logging.add_line(tweet.text)
                        else:
                            if(len(tweet.retweeted_by_ids) > 0):
                                tweet_id = tweet.retweeted_by_ids[0]
                                tweet = glb.glb_dict_tweets[tweet_id]
                                logging.add_line(tweet.text)
#                logging.add_items(items)
            if glb.gsr.has_key(dt.date()):
                logging.add_line('{event}')
            else:
                logging.add_line('{no event}')
            

def test_hashtag_stats():

    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[0:1]:
        print co_name
        log_folder = os.path.join('D:/embers/stat', co_name)
        logging = Log(log_folder + '/hashtag_evaluations_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
#        logging_sum = Log(log_folder + '/evaluations_summary_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    #    return 
        desps = ['# tweets']
        logging.add_line('{}'.format(desps))
        for di in range(60):
            if di > len(files) - 1:
                continue
    #        agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
            print files[di]
            dt = datetime.strptime(files[di], "%Y-%m-%d")
            logging.add_empty_lines(4)
            logging.add_line('************************************')
            logging.add_line(files[di])
            for i in range(1):
#                logging.add_line(desps[i])
                items = []
                for key, hashtag in glb.glb_dict_hashtags.items():
                    hashtag.calc_features(glb)
                    items.append((key, hashtag.features[i]))
                logging.add_empty_lines(2)
                cp_items = copy.deepcopy(items)
                cp_items.sort(key = lambda x: x[1], reverse = True)
                values = [val[1] for val in cp_items]
                
                logging.add_line('{}: {}, count ({}), mean(std): {} ({}), median(mad): {} ({})'.format(desps[i], values[0:10], len(items), mean(values), std(values), median(values), median([abs(val - median(values)) for val in values])))

                if i == 0:
                    for j in range(10):
                        if j >= len(cp_items):
                            continue
                        key = cp_items[j][0]
                        hashtag = glb.glb_dict_hashtags[key]
                        logging.add_line('****************************: {}'.format(hashtag.name))
                        for tweet_id in hashtag.dict_tweet_ids.keys()[0:5]:
                            tweet = glb.glb_dict_tweets[tweet_id]
                            logging.add_line(tweet.text)
                            
#                logging.add_items(items)
            if glb.gsr.has_key(dt.date()):
                logging.add_line('{event}')
            else:
                logging.add_line('{no event}')


def test_hashtag_stats_1():

    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[0:1]:
        print co_name
        log_folder = os.path.join('D:/embers/stat', co_name)
        logging = Log(log_folder + '/hashtag_evaluations_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        excel_logging = Log(log_folder + '/excel_hashtag_evaluations_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.csv')
#        logging_sum = Log(log_folder + '/evaluations_summary_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    #    return 
        hashtag_dict = dict()
        desps = ['# tweets']
        logging.add_line('{}'.format(desps))
        event_labels = []
        for di in range(30):
            if di > len(files) - 1:
                continue
            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
            print files[di]
            dt = datetime.strptime(files[di], "%Y-%m-%d")
            logging.add_empty_lines(4)
            logging.add_line('************************************')
            logging.add_line(files[di])
            for i in range(1):
                for key, hashtag in glb.glb_dict_hashtags.items():
                    hashtag.calc_features(glb)
                    if hashtag_dict.has_key(key) == False:
                        hashtag_dict[key] = [0 for j in range(di)] + [hashtag.features[i]]
                    else:
                        hashtag_dict[key].append(hashtag.features[i])
                for key, value in hashtag_dict.items():
                    if len(hashtag_dict[key]) == di:
                        hashtag_dict[key].append(0)
                    elif len(hashtag_dict[key]) != di + 1:
                        warnings.warn('something wrong: hashtag_dict[key].append(hashtag.features[i]): {}, {}'.format(len(hashtag_dict[key]), di+1))
                        
            if glb.gsr.has_key(dt.date()):
                event_labels.append(1)
            else:
                event_labels.append(0)

        sort_hashtag_stat = hashtag_dict.items()
        sort_hashtag_stat = sorted(sort_hashtag_stat, key = lambda x: sum(x[1]) * -1)
        
        excel_logging.add_line(',,,,,{}'.format(','.join(map(str, event_labels))))
        for (key, values) in sort_hashtag_stat:
#            hashtag = glb.glb_dict_hashtags[key]
            logging.add_line('****************************: {}'.format(key))
            logging.add_line('count ({}), mean(std): {} ({}), median(mad): {} ({})'.format(len(values), mean(values), std(values), median(values), median([abs(val - median(values)) for val in values])))
            logging.add_line('{}'.format(values))
            excel_logging.add_line('{},{},{},{},{},{}'.format(key, mean(values), std(values), median(values), median([abs(val - median(values)) for val in values]), ','.join(map(str, values))))
                    
                
                
                
def test_keyword_stats():
    
    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[0:1]:
        print co_name
        log_folder = os.path.join('D:/embers/stat', co_name)
        logging = Log(log_folder + '/evaluations_maxdata_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
#        logging_sum = Log(log_folder + '/evaluations_summary_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt')
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    #    return 
        desps = ['# tweets', '# retweets', '# followers', '# followees', '# mentioned_by', 'E replied_by', 'diff graph depth', 'diff graph size']
        logging.add_line('{}'.format(desps))
        for di in range(30):
            if di > len(files) - 1:
                continue
    #        agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
            print files[di]
            logging.add_empty_lines(4)
            logging.add_line('************************************')
            logging.add_line(files[di])
            items = []
            for key, keyword in glb.glb_dict_city_keywords.items():
                latlong = key[0]
                kw_name = key[1]
                city = glb.glb_dict_cities[latlong]
                
                if len(city.dict_tweet_ids) > 5:
                    count = len(keyword.dict_tweet_ids)
                    if count > 10:
                        items.append((keyword.ci_name, keyword.df_idf, count, keyword.ori_kw))
            items = sorted(items, key=lambda x: x[2] * -1)
            logging.add_line('Sorting based on count')
            logging.add_line('{}'.format(items[0:50]))

            items = sorted(items, key=lambda x: x[1] * -1)
            logging.add_line('Sorting based on IDF')
            logging.add_line('{}'.format(items[0:50]))


"""
Calculate for each city, the min, max, average, media count of each term. 
"""
def calc_keyword_stats():
    
    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
    co_name = countries[1]
    for co_name in countries[0:1]:
        print co_name
        raw_twitter_folder   = os.path.join(root, 'data')
    
        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
    
        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
        n_files = len(files)
#        n_files = 2
#
#        glb_dict_city_keywords = dict()
#        for di in range(n_files):
#            if di > len(files) - 1:
#                continue
#            print files[di]
#            data_file = os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl')
#            if os.path.exists(data_file):
#                glb = Share.load_pkl_file(data_file)
#                for key, keyword in glb.glb_dict_city_keywords.items():
#    #                city = glb.glb_dict_cities[latlong]
#                    if glb_dict_city_keywords.has_key(key) == False:
#                        glb_dict_city_keywords[key] = [keyword]
#                    else:
#                        glb_dict_city_keywords[key].append(keyword)
#                
#        Share.save_to_pkl_file(glb_dict_city_keywords, 'glb_dict_city_keywords.pkl')

        vocabulary = dict()
        glb_dict_city_keywords = Share.load_pkl_file('glb_dict_city_keywords.pkl')
        glb_dict_city_keywords_stat = dict()
        for key, elements in glb_dict_city_keywords.items():
            counts = [len(element.dict_tweet_ids) for element in elements]
            for i in range(n_files - len(counts)):
                counts.append(0)
            stat = [key[1], len(counts), mean(counts), std(counts), median(counts), median([abs(val - median(counts)) for val in counts])]
            vocabulary[element.ori_kw] = 1
            if len(counts) > 10 and stat[2] + 3 * stat[3] > 10:
#                print '{}: {}'.format(element.ori_kw, counts)
                print stat
            glb_dict_city_keywords_stat[key] = stat
            
#        print vocabulary
#        print len(vocabulary)
        
        Share.save_to_pkl_file(glb_dict_city_keywords, 'glb_dict_city_keywords_stat.pkl')


        
def experiment():
    
    test_hashtag_stats_1()
    
    return 
    
#    lines = open('D:/embers/data/usersLists/twitter_users_list/usersLists/usersMexico/followerGraphMexico-0621-0831.adj.dir.txt').readlines()
#    print lines[0:100]
#    
#    return 
    
#    print datetime.now()
#    data = Share.load_pkl_file('D:/embers/stat/mexico/city_term_freq_primal.pkl')
#    print datetime.now()
#    print size(data.items())
#    print datetime.now()
#    print data.items()[0:200]
#    print datetime.now()
#    
##    test_tweet_stats()
#    
##    calc_keyword_stats()
#    
##    test_keyword_stats()
#    
##    test_user_stats()
#
#    return     
###    line = open('D:\embers\data\mexico/filtered-2012-10-1.txt').readlines()[0]
##    json.load(line)
##    data = json.load(open('D:/embers/data/mexico/filtered-2012-9-29.txt'), encoding='shift_jis')
##    
##    glb = Share.load_pkl_file('D:/embers/stat/mexico/glb_2012-06-25.txttweets2items.txt.pkl')
##    agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/backup-3-5-13/agg_stats.pkl')
##    agg_stats = Share.load_pkl_file('D:/embers/stat/chile/agg_stats_test.pkl')
##    day_stat = Share.load_pkl_file('D:/embers/stat/chile/day_stat.pkl')
#    import matplotlib.pyplot as plt
#
#    countries = ['mexico', 'argentina', 'brazil', 'chile', 'ecuador', 'paraguay', 'venezuela', 'colombia']
#    co_name = countries[1]
#    for co_name in countries[2:]:
#        log_folder = os.path.join('D:/embers/stat', co_name)
#        logging = Log(log_folder + '/evaluations.txt')
#        raw_twitter_folder   = os.path.join(root, 'data')
#    
#        files = [f for f in os.listdir(raw_twitter_folder + '/' + co_name) if f.endswith('.txt')]
#        files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
#    
#        files = [f.replace('filtered-', '').replace('.txt', '') for f in files]
#    #    files.sort(key = lambda x: datetime.strptime(x.replace('filtered-', '').replace('.txt', ''), "%Y-%m-%d"))
#    #    return 
#        for di in range(120):
#    #        agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
#            glb = Share.load_pkl_file(os.path.join('D:/embers/stat', co_name, 'glb_filtered-' + files[di] + '.txt.pkl'))
#    #        print type(glb.gsr.keys()[0])
#    #        list_dts = sort(glb.gsr.keys())
#    #        for dt in list_dts:
#    #            print dt
#    #        continue
#            print files[di]
#            logging.add_line(files[di])
#            for i in range(8):
#                items = []
#                for key, user in glb.glb_dict_users.items():
#                    items.append(user.calc_features(glb)[i])
#                    if user.features[7] == 28:
#                        logging.add_line(user.get_basic_info())
#                        for kt, tweet_id in user.dict_tweet_ids.items():
#                            tweet = glb.glb_dict_tweets[kt]
#        #                    for kw, kw_id in tweet.dict_keywords.items():
#        #                        print kw
#                            logging.add_line(tweet.text)
#                logging.add_line('max value: {}'.format(max(items)))
#                logging.add_line('{}'.format(items))
#            for key, link in glb.glb_dict_links.items():
#        #        print link.http
#                if link.http == 'http://t.co/bfHiXukR':
#                    logging.add_line('matched')
#                    logging.add_line('{}'.format(len(link.link_tweet_ids)))
##    print items
##    plt.hist(items)
##    plt.show()        
##    glb = Share.load_pkl_file('D:/embers/stat/mexico/glb_filtered-2012-6-25.txt.pkl')
##    mean_std_stat = Share.load_pkl_file('D:/embers/stat/mexico/mean_std_stat.pkl')
##    print mean_std_stat.dict_country_keywords_threshold[mean_std_stat.dict_country_keywords_threshold.keys()[1]]
#    
##    agg_stats = Share.load_pkl_file('D:/embers/stat/mexico/agg_stats.pkl')
##    b1 = agg_stats.dict_country_keywords_stat[agg_stats.dict_country_keywords_stat.keys()[0]]
##    b2 = agg_stats.dict_country_keywords_dates[agg_stats.dict_country_keywords_dates.keys()[0]]
##    gsr = agg_stats.gsr
##    re = Stats.calc_feature_threshold(b1, b2, gsr)
##    print re
#
#    test = 1
#    return 
##    
#    all_gsr = Share.load_pkl_file('D:/embers/data/upd_events_count.pkl')    
#    return 
    
    num_consumers = 1
    multicore_proc(num_consumers)

    
def main():
        
    experiment()


if __name__ == '__main__':
    main()



