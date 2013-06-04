# -*- coding: utf-8 -*-
import gzip
import os
import json
import nltk
import multiprocessing
import datetime
import sys
from generic_funs import gen_funs

# Description: The function is to stem the tweet text
# Paramteter: content is the text of each raw tweet
def stem_text(content):
    tokens = nltk.word_tokenize(content)
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    words = [w.lower().strip() for w in tokens if w not in [",",".",")","]","(","[","*",";","...",":","&",'"',"'","’"] and not w.isdigit()]
    words = [w for w in words if w.encode("utf-8") not in nltk.corpus.stopwords.words('english')]
    stemmedWords = [stemmer.stem(w) for w in words]
    return stemmedWords

def stem_word(w):
#    tokens = nltk.word_tokenize(content)
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    return stemmer.stem(w)
    
words = open('transwords.txt').readlines()
words = [word.lower().strip() for word in words]
#print words 

keyWordDict={}
englishCountries={}
us_time_zones={'arizona':1,'alaska':2,'central':3,'eastern':4,'hawaii':5,'mountain':6,'pacific':7}

keyWordObject = open("KWordsList1.txt","r")
keyWords = keyWordObject.read().split('\t')
keyWords = keyWords + words
#keyWords = words
i=0
for keyWord in keyWords:
    keyWordDict[stem_word(keyWord.strip().lower())]=i
    i=i+1

#funs.json_save('transportation_keywords.json', keyWordDict.keys())

keyWordDict['road'] = 1
keyWordDict['traffic'] = 1
keyWordDict['highway'] = 1
keyWordDict['transit'] = 1
keyWordDict['accident'] = 1
keyWordDict['car'] = 1
keyWordDict['street'] = 1
keyWordDict['signal'] = 1
keyWordDict['lane'] = 1


#print keyWordDict

keyWordObject = open("EnglishCountryList.json","r")
keyWords = keyWordObject.read().split('\n')
i=0
for keyWord in keyWords:
    countryName =keyWord.decode('ascii', 'ignore')
    englishCountries[countryName.strip().lower()]=i
    i=i+1

#print englishCountries

"""
    check if tweet is legal format
"""
def is_valid_tweet_format(tweet):
    if tweet.has_key('user'):
        return True
    else:
        return False

"""
    Return true if the tweet user's time_zone is U.S. timezone. Note that if the tweet does not provide any time_zone info, we still return true. We only return false, if it has explict time zone info that is not U.S. type
"""
def is_us_time_zone(tweet):
    timezone_matched = False
    if tweet['user'].has_key('time_zone'):
        if tweet['user']['time_zone']:
            tweet_timezone = tweet['user']['time_zone'].lower()
            for timezone in us_time_zones.keys():
                if tweet_timezone.find(timezone) >= 0:
                    timezone_matched = True         
                    break
    return timezone_matched

def is_eng(tweet):
    if tweet.has_key('lang'):
        if tweet['lang']=="en":
            return True
    if tweet.has_key('user') and tweet['user'].has_key('lang'):
        if tweet['user']['lang']=="en":
            return True
    return False

def is_us_location(tweet):
    within_range = True
    if tweet.has_key('user') and tweet['user'].has_key('location'):
        loc_text = tweet['user']['location'].lower().strip()
#        print loc_text
        for co in englishCountries.keys():
            if loc_text.find(co) >= 0:
                within_range = False
                break
    return within_range
    
class Task_stats_calc(object):
    def __init__(self, input_folder, str_dt, output_folder):
        self.input_folder = input_folder
        self.str_dt = str_dt
        self.output_folder = output_folder
        
    def __call__(self):

        output_file_path = self.output_folder+"/"+self.str_dt + '.txt'
        if os.path.exists(output_file_path):
            return
            
        raw_tweet_files = [rawFile for rawFile in os.listdir(self.input_folder) if rawFile.endswith('.gz') and rawFile.find(self.str_dt) >= 0]
#        print 'check files:'        
#        print raw_tweet_files
#        return 
        rawTweetFiltered = []
#        rawTwitterFiltered = gzip.open(output_file_path, 'wb')
        for rawTwitterFile in raw_tweet_files[0:2]:
            print rawTwitterFile
            rawTwitterData = gzip.open(os.path.join(self.input_folder, rawTwitterFile),'rb')
            try:
                file_content=rawTwitterData.readlines()
            except:
                print '!!!!!!!!!!!!!!!!!!! file procesing error: {}'.format(rawTwitterFile)
                file_content = None
                pass
            if file_content:
                for line in file_content:
                    try:
#                        print line
                        tweet  = json.loads(line)
                        if is_valid_tweet_format(tweet) and is_us_time_zone(tweet) and is_eng(tweet) and is_us_location(tweet):
#                            print '############### passed'
                            tweetWords = stem_text(tweet["text"])
                            if check_istransportation_tweet(tweetWords) or check_isTop25Account(tweet):
#                                print '############### passed @@@@@@@@'
#                                sys.stdout.flush()
                                rawTweet={}
                                rawTweet["created_at"]=tweet["created_at"]
                                if tweet.has_key('hashtags'):
                                    rawTweet["hashtags"]=tweet['hashtags']
                                elif tweet.has_key('entities'):
                                    if tweet['entities'].has_key('hashtags'):
                                        rawTweet["hashtags"]=tweet['entities']['hashtags']
                                rawTweet["name"]=tweet["user"]["name"]
                                rawTweet["user_id"]=tweet["user"]["id"]
                                rawTweet["location"]=tweet["user"]["location"]
                                rawTweet["text"]=tweet["text"]
                                rawTweet["id"]=tweet["id"]
                                rawTweetFiltered.append(tweet)
                    except:
                        pass
                print '{}: totally {} tweets processed'.format(rawTwitterFile, len(rawTweetFiltered))
                sys.stdout.flush()
        #gen_funs.new_folder(output_file_path)
        gen_funs.json_save(output_file_path, rawTweetFiltered)
    def __str__(self):
        return '%s processed!' % (self.str_dt)
        

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                print '%s: Exiting' % proc_name
                break
            print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return
        
# Description: The function is to retrieve the key words of transportation dictionary
# Paramteter: linksFile is to define the dictionary file path
#def read_filter_dictionary():

# Descrption: The function
def read_top25_tweet_accounts():
    tweetAccounts=[{'name':'Matt Yglesias','twitterName':'@mattyglesias'},{'name':'Richard Florida','twitterName':'@Richard_Florida'},\
                   {'name':'The Atlantic Cities','twitterName':'@AtlanticCities'},{'name':'Reconnecting America','twitterName':'@reconnecting'},\
                   {'name':'Ray LaHood','twitterName':'@RayLaHood'},{'name':'Mark Abraham','twitterName':'@urbandata'},\
                   {'name':'Urban Land Institute','twitterName':'@UrbanLandInst'},{'name':'Planetizen','twitterName':'@planetizen'},\
                   {'name':'Curbed – New York','twitterName':'@curbedNY'},{'name':'WNYC’s Transportation Nation','twitterName':'@transportnation'},\
                   {'name':'Aaron Renn','twitterName':'@urbanophile'},{'name':'Project for Public Spaces','twitterName':'@PPS_Placemaking'},\
                   {'name':'Kaid Benfield','twitterName':'@Kaid_at_NRDC '},{'name':'Streetsblog Network','twitterName':'@StreetsblogNet'},\
                   {'name':'The Infrastructurist','twitterName':'@Infrastructurst'},{'name':'Congress for the New Urbanism','twitterName':'@NewUrbanism'},\
                   {'name':'Nate Berg','twitterName':'@nate_berg'},{'name':'Yonah Freemark','twitterName':'@Ttpolitic  '},\
                   {'name':'Transport Data','twitterName':'@transportdata'},{'name':'Midwest High Speed Rail Association','twitterName':'@HSRail'},\
                   {'name':'Transportation for America','twitterName':'@T4America'},{'name':'Smart Growth America','twitterName':'@SmartGrowthUSA'},\
                   {'name':'American Planning Association','twitterName':'@APA_Planning'},\
                   {'name':'National Complete Streets Coalition','twitterName':'@completestreets '},\
                   {'name':'Building America’s Future Educational Fund','twitterName':'@BAFuture'}]
    return tweetAccounts    



#Description: The function is to check whether the stemmed words has the transportation key word.
#Parameter: stemmedWords is the stemmed words list from the stem_text function
def check_istransportation_tweet(stemmedWords):
    for word in stemmedWords:
        if keyWordDict.has_key(word.strip().lower()):
            return True
    return False

#Description: The function is to check whether the tweet has something with Top 25 tweet accounts. If it is related, return True; or else, False.
#Parameter: tweet is the orginial raw tweet data.
def check_isTop25Account(tweet):
    tweetAccountsTop25 = read_top25_tweet_accounts()
    dic={}
    dic.itervalues()
    for tweetAccount in tweetAccountsTop25:
        for tweetName in tweetAccount.itervalues():
            tweetName = tweetName.decode('utf8')
            if tweet['user']['name'].lower()==tweetName.lower():
                return True
            if tweet['user']['screen_name'].lower()==tweetName.replace('@','').lower():
                return True
            if tweetName.lower().strip() in tweet['text'].lower():
                return True
    return False

def main():
    
    input_folder = 'G:/TwitterData_Dpbx/2013-02'
#    input_folder = 'J:\Python\ICDM\input'
    output_folder = 'G:/TwitterData_Dpbx/1/filtered'
#    output_folder = 'J:\Python\ICDM\output'    
    raw_tweet_files = [rawFile for rawFile in os.listdir(input_folder) if rawFile.endswith('.gz')]
    
    dict_str_dt = dict()
    for filename in raw_tweet_files[0:2]:
        filename = filename.replace('tweets.', '')
        idx = filename.find('T')
        str_dt = filename[0:idx]
        dict_str_dt[str_dt] = 1
    
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()

    num_consumers = 8
    
    # Start consumers
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    
    num_jobs = len(dict_str_dt.keys())

    # Enqueue jobs
    for str_dt in dict_str_dt.keys()[0:num_jobs]:
        print str_dt
        tasks.put(Task_stats_calc(input_folder, str_dt, output_folder))

    # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)
        
    while num_jobs:
        results.get()
        print 'num jobs: ', num_jobs
        num_jobs -= 1
        
if __name__ == "__main__":
    main()