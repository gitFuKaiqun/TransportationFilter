from numpy.random import *
from numpy import *
from datetime import *
from operator import itemgetter
import csv
import re
import os
import json
import copy
import pickle
import warnings
import multiprocessing
import sys
from scipy.spatial import KDTree
from geocode2 import Geo2
import pylab
from nltk import SnowballStemmer
from feature_extraction.text import FeatureCountVectorizer
from Geographic_location import Geographic_location

glb_geo = Geo2()


from Graph import Graph
from Anomalies import Anomalies
from Log import Log
from Follow_graph import Follow_graph
from cls_emp_dist import cls_emp_dist
from Link import Link
from Hashtag import Hashtag
from Keyword import Keyword
from Geographic_location import Geographic_location
from Country import Country
from State import State
from City import City
from diffusion_graph import diffusion_graph
from reply_graph import reply_graph
from retweet_graph import retweet_graph

"""
Created on Tue Feb 05 22:08:44 2013

@author: Feng Chen
"""
#latlong_2_city = pickle.load(open(root + 'data/latlong2citycountry.pkl'))
WG_FIELDS = ["id", "name", "alt_names", "orig_names", "type", "pop",
             "longitude",
             "latitude",
             "country", "admin1", "admin2", "admin3"]

COUNTRY_STRS = ("Argentina, Brazil, Honduras, Colombia, Venezuela, Peru, "
                "Chile, Panama, Belize, Bolivia, Uruguay, Paraguay, "
                "Nicaragua, Suriname, Costa Rica, El Salvador, Ecuador, "
                "Guyana, French Guiana, Guatemala, Mexico")
CITY_DATA = os.path.join(os.path.dirname(__file__), "data/city_data.txt")
WG_DATA   = os.path.join(os.path.dirname(__file__), "data/wg-partial.txt")

#line = open('D:\embers\data\mexico/filtered-2012-10-1.txt').readlines()[0]

glb_cdf_values = [0.5, 0.5039893563146316, 0.50797831371690194, 0.51196647341411261, 0.51595343685283079, 0.51993880583837249, 0.52392218265410684, 0.52790317018052113, 0.53188137201398744, 0.53585639258517215, 0.53982783727702899, 0.54379531254231683, 0.54775842602058389, 0.55171678665456114, 0.55567000480590645, 0.5596176923702425, 0.56355946289143288, 0.56749493167503839, 0.57142371590090069, 0.57534543473479549, 0.57925970943910299, 0.58316616348244232, 0.58706442264821457, 0.59095411514200591, 0.59483487169779581, 0.5987063256829237, 0.60256811320176051, 0.60641987319803958, 0.61026124755579725, 0.61409188119887737, 0.61791142218895256, 0.62171952182201928, 0.62551583472332006, 0.62930001894065357, 0.63307173603602807, 0.6368306511756191, 0.64057643321799129, 0.64430875480054672, 0.64802729242416279, 0.65173172653598233, 0.65542174161032418, 0.65909702622767741, 0.66275727315175048, 0.66640217940454227, 0.67003144633940637, 0.67364477971208003, 0.67724188974965227, 0.6808224912174442, 0.68438630348377738, 0.68793305058260945, 0.69146246127401312, 0.69497426910248061, 0.69846821245303381, 0.70194403460512356, 0.70540148378430201, 0.70884031321165364, 0.71226028115097295, 0.71566115095367588, 0.71904269110143559, 0.72240467524653507, 0.72574688224992645, 0.72906909621699434, 0.732371106531017, 0.73565270788432247, 0.73891370030713843, 0.74215388919413527, 0.74537308532866386, 0.74857110490468992, 0.75174776954642941, 0.75490290632569057, 0.75803634777692697, 0.76114793191001329, 0.76423750222074882, 0.76730490769910253, 0.77035000283520949, 0.77337264762313174, 0.77637270756240062, 0.77935005365735033, 0.78230456241426682, 0.78523611583636288, 0.78814460141660336, 0.79102991212839835, 0.79389194641418692, 0.79673060817193164, 0.79954580673955034, 0.80233745687730762, 0.80510547874819161, 0.80784979789630385, 0.81057034522328786, 0.81326705696282731, 0.81593987465324047, 0.81858874510820279, 0.82121362038562828, 0.82381445775474216, 0.82639121966137541, 0.82894387369151823, 0.83147239253316219, 0.83397675393647042, 0.83645694067230769, 0.83891294048916909, 0.84134474606854293, 0.84375235497874534, 0.84613576962726522, 0.84849499721165633, 0.85083004966901865, 0.85314094362410409, 0.85542770033609039, 0.85769034564406077, 0.85992890991123105, 0.8621434279679645, 0.86433393905361733, 0.86650048675725277, 0.86864311895726931, 0.8707618877599822, 0.87285684943720176, 0.87492806436284976, 0.87697559694865657, 0.87899951557898182, 0.88099989254479927, 0.88297680397689127, 0.88493032977829178, 0.88686055355602256, 0.88876756255216538, 0.89065144757430814, 0.89251230292541306, 0.89435022633314465, 0.89616531887869955, 0.89795768492518091, 0.89972743204555794, 0.90147467095025213, 0.9031995154143897, 0.90490208220476098, 0.90658249100652821, 0.90824086434971929, 0.90987732753554762, 0.91149200856259804, 0.91308503805291497, 0.91465654917803296, 0.91620667758498575, 0.91773556132233103, 0.91924334076622893, 0.92073015854660767, 0.92219615947345357, 0.92364149046326083, 0.92506630046567295, 0.9264707403903516, 0.92785496303410619, 0.92921912300831444, 0.93056337666666833, 0.93188788203327455, 0.93319279873114191, 0.93447828791108356, 0.93574451218106414, 0.93699163553602161, 0.93821982328818809, 0.93942924199794098, 0.94062005940520699, 0.94179244436144705, 0.94294656676224586, 0.94408259748053058, 0.94520070830044201, 0.94630107185188028, 0.94738386154574794, 0.94844925150991066, 0.94949741652589625, 0.9505285319663519, 0.95154277373327723, 0.95254031819705265, 0.95352134213627993, 0.95448602267845017, 0.95543453724145699, 0.95636706347596812, 0.95728377920867103, 0.9581848623864051, 0.95907049102119268, 0.95994084313618289, 0.96079609671251731, 0.96163642963712881, 0.96246201965148326, 0.9632730443012737, 0.96406968088707423, 0.9648521064159612, 0.96562049755411006, 0.96637503058037166, 0.96711588134083615, 0.96784322520438626, 0.96855723701924734, 0.96925809107053407, 0.96994596103880026, 0.9706210199595906, 0.97128344018399826, 0.97193339334022755, 0.9725710502961632, 0.97319658112294505, 0.97381015505954727, 0.97441194047836144, 0.97500210485177952, 0.97558081471977742, 0.97614823565849151, 0.97670453224978815, 0.97724986805182079, 0.97778440557056856, 0.97830830623235321, 0.97882173035732778, 0.97932483713392993, 0.97981778459429558, 0.98030072959062309, 0.98077382777248268, 0.98123723356506221, 0.98169110014834104, 0.98213557943718344, 0.98257082206234292, 0.98299697735236724, 0.98341419331639501, 0.98382261662783388, 0.98422239260890954, 0.98461366521607452, 0.98499657702626775, 0.98537126922401075, 0.98573788158933118, 0.98609655248650141, 0.98644741885358, 0.98679061619274377, 0.98712627856139801, 0.98745453856405341, 0.98777552734495533, 0.98808937458145296, 0.98839620847809651, 0.9886961557614472, 0.98898934167558861, 0.98927588997832416, 0.98955592293804895, 0.98982956133128031, 0.99009692444083575, 0.99035813005464168, 0.99061329446516144, 0.99086253246942735, 0.99110595736966323, 0.99134368097448344, 0.99157581360065428, 0.99180246407540384, 0.99202373973926627, 0.99223974644944635, 0.99245058858369084, 0.99265636904465171, 0.99285718926472855, 0.99305314921137566, 0.99324434739285938, 0.99343088086445319, 0.99361284523505677, 0.99379033467422384, 0.9939634419195873, 0.99413225828466745, 0.99429687366704933, 0.99445737655691735, 0.99461385404593328, 0.99476639183644422, 0.994915074251009, 0.9950599842422293, 0.99520120340287377, 0.99533881197628127, 0.99547288886703267, 0.99560351165187866, 0.9957307565909107, 0.99585469863896392, 0.99597541145724167, 0.99609296742514719, 0.99620743765231456, 0.99631889199082502, 0.99642739904760025, 0.99653302619695938, 0.9966358395933308, 0.99673590418410873, 0.99683328372264224, 0.99692804078134956, 0.99702023676494544, 0.99710993192377384, 0.99719718536723501, 0.99728205507729872, 0.99736459792209509, 0.99744486966957202, 0.99752292500121409, 0.9975988175258107, 0.9976725997932685, 0.99774432330845764, 0.99781403854508677, 0.99788179495959539, 0.99794764100506028, 0.99801162414510569, 0.99807379086781212, 0.99813418669961596, 0.99819285621919362, 0.99824984307132392, 0.99830518998072271, 0.99835893876584303, 0.99841113035263518, 0.99846180478826196, 0.99851100125476255, 0.99855875808266004, 0.99860511276450781, 0.9986501019683699, 0.99869376155123057, 0.99873612657232769, 0.99877723130640772, 0.9988171092568956, 0.99885579316897732, 0.99889331504259071, 0.99892970614532106, 0.99896499702519714, 0.99899921752338594, 0.99903239678678168, 0.99906456328048587, 0.99909574480017771, 0.99912596848436841, 0.99915526082654138, 0.99918364768717138, 0.99921115430562446, 0.99923780531193274, 0.9992636247384461, 0.99928863603135465, 0.99931286206208414, 0.99933632513856008, 0.99935904701633993, 0.99938104890961321, 0.99940235150206558, 0.99942297495760923, 0.99944293893097536, 0.99946226257817028, 0.99948096456679303, 0.99949906308621428, 0.99951657585761622, 0.99953352014389241, 0.99954991275940785, 0.99956577007961833, 0.99958110805054967, 0.99959594219813597, 0.99961028763741799, 0.99962415908159996, 0.99963757085096694, 0.99965053688166206, 0.99966307073432314, 0.99967518560258117, 0.99968689432141877, 0.99969820937539133, 0.9997091429067092, 0.99971970672318378, 0.99972991230603647, 0.99973977081757248, 0.99974929310871952, 0.99975848972643211, 0.99976737092096446, 0.99977594665300895, 0.99978422660070532, 0.99979222016651936, 0.99979993648399268, 0.99980738442436434, 0.99981457260306672, 0.99982150938609515, 0.99982820289625407, 0.99983466101927987, 0.99984089140984245, 0.99984690149742628, 0.99985269849209257, 0.99985828939012422, 0.99986368097955425, 0.99986887984557948, 0.99987389237586155, 0.9998787247657146, 0.99988338302318458, 0.99988787297401771, 0.99989220026652259, 0.99989637037632595, 0.99990038861102404, 0.9999042601147311, 0.99990798987252594, 0.99991158271479919, 0.99991504332150205, 0.99991837622629731, 0.99992158582061641, 0.99992467635762128, 0.99992765195607491, 0.99993051660412013, 0.99993327416297029, 0.99993592837051115, 0.99993848284481679, 0.99994094108758103, 0.99994330648746577, 0.99994558232336628, 0.99994777176759819, 0.9999498778890038, 0.99995190365598241, 0.99995385193944375, 0.9999557255156879, 0.99995752706921126, 0.99995925919544149, 0.99996092440340223, 0.99996252511830896, 0.99996406368409718, 0.99996554236588497, 0.99996696335237056]
TIME_FORMAT = "%a, %d %b %Y %H:%M:%S +0000"

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

class funs(object):

    """
        OUTPUT: 
            flt_co_gsr: [dt, loc, desp, first_report_link, embersid], where loc: (co, st, ci)
    """
    @staticmethod
    def flt_gsr_by_cos(co_name):
        
        countries = [co_name]
        flt_event_types = ['0111', '0112', '0121', '0122', '0131', '0132', '0141', '0142', '0151', '0152', '0161', '0162', '0171', '0172']
        flt_event_types_desp = ['Employment-Non-Violent', 'Employment-Violent', 'Housing-Non-Violent', 'Husing-Violent', 'Energy&Resurces-Non-Violent', 'Energy&Resurces-Violent', 'Other-Economic-Polices-Non-Violent', 'Other-Economic-Polices-Violent', 'Other-Government-Policies-Non-Violent', 'Other-Government-Policies-Violent', 'Other-Non-Violent', 'Other-Violent', 'Unspecified-Non-Violent', 'Unspecified-Violent']
        
        dict_event_types_desp = dict()
        for etype, edesp in zip(flt_event_types, flt_event_types_desp):
            dict_event_types_desp[etype] = edesp
    
        flt_co_gsr = []
        f = 'C:/Users/Heinz/Dropbox/EMBERS/FengTweeter/project_code/scratch-master/civilUnrest/data/all_gsr_warnings.json'
        for line in open(f).readlines():
            event = json.loads(line)
            event_type = event['eventType']
#            if event_type in flt_event_types and dict_event_types_desp[event_type].find('Non-Vio') < 0:
            if event_type in flt_event_types:
                loc = event['location']
                desp = event['derivedFrom']['description']
#                if desp.find('ILI EW Case Count') > 0:
#                    continue
                first_report_link = event['derivedFrom']['firstReportedLink']
                embersid = event['embersId']
                dt_str = event['eventDate']
                dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
                # country, state, city
                loc[0] = loc[0].lower()
                loc[1] = loc[1].lower()
                loc[2] = loc[2].lower()
                loc = tuple(loc)
                
                if loc[0] not in countries:
                    continue
                
                flt_co_gsr.append([dt, loc, desp, first_report_link, embersid])
                
            else:
                pass
    
        return flt_co_gsr

    @staticmethod
    def proc_twitter_merge1(in_root, out_root, co_name):
        
        dict_civil_keyword = funs.load_pkl_file('data/dict_civil_keyword.pkl')
        in_folder = os.path.join(in_root, co_name)
        out_folder = os.path.join(out_root, co_name)
        enrich_files = [f for f in os.listdir(in_folder) if f.endswith('.txt')]
        enrich_files.sort(key = lambda x: datetime.strptime(x.replace('.txttweets2items', '').replace('.txt', ''), "%Y-%m-%d"))
    
        new_enrich_files = []
        for f in enrich_files:
            if os.path.exists(out_folder + '/' + f):
                pass
            else:
                new_enrich_files.append(f)
        
        enrich_files = new_enrich_files
        
        for efile in enrich_files:
            print efile
            merge_data = []
            for line in open(os.path.join(in_folder, efile)).readlines():
                tweets = json.loads(line)
#                print tweet
#                print tweet['text_items']
                for tweet in tweets:
                    items = tweet['text_items'][0]
                    for item in tweet['text_items'][1]:
                        items.append(item.replace('#', ''))
                    n_matchs = 0
                    for item in items:
                        if dict_civil_keyword.has_key(item):
                            n_matchs += 1
                    if n_matchs >= 2:
                        merge_data.append(tweet)
            funs.new_folder(out_folder)
            funs.json_save(os.path.join(out_folder, efile), merge_data)

#    @staticmethod
#    def format_gsr_loc(location):                       
#        co, a, ci = utils_new.osi_capital_city_province_corrector(location)
#        if a is '-':
#            a1 = None
#        else:
#            a1 = a
#        print 'start'
#        print ci,a1, co
#        cor_loc = funs.loc(geo2.best_guess(ci,a1, co))
#        print cor_loc
#        if cor_loc:
#            res = utils_new.osi_capital_city_province_corrector(cor_loc)
#        else:
#            print location
#            return tuple(location)
#        print res
#    #    flag = True
#        
#        return (res[0].lower(), res[1].lower(), res[2].lower())
#
#    @staticmethod        
#    def loc(record):
#        if record != None and len(record[0]) > 0:
#            return (record[0][0][1], record[0][0][2], record[0][0][0])
#        else:
#            return None

    @staticmethod
    def mergdata(folder):
    
        raw_tweet_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        for in_filename in raw_tweet_files:
            print in_filename
            json_data_path = folder + '/' + in_filename
            json_data = open(json_data_path, 'r')
#            print json_data
            try:
                data = json.load(json_data, encoding='utf-8')
            except:
                print json_data
                sys.stdout.flush()
                print 'this is a test'
                line = open(json_data_path).readlines()[0]
                items = line.split('}][{')
                agg_data = []
                for idx, item in enumerate(items):
                    if idx == 0:
                        temp = json.loads(item + '}]')
                    else:
                        if idx < len(items) - 1:
                            try:
                             temp = json.loads('[{' + item + '}]')
                            except:
                                print item
                                temp = json.loads('[{' + item + '}]')
                        else:
                            try:
                                temp = json.loads('[{' + item)
                            except:
                                print item
                                temp = json.loads('[{' + item)
                    agg_data.extend(temp)
                data = agg_data
                f_out = open(json_data_path, "w")
                f_out.write(json.dumps(data))
                f_out.close()


    
    @staticmethod
    def day_hetergenegous_graph_generation(day_raw_tweets, glb, root, co_name): 
    
        glb.load_data(root, co_name) 
        # Identify keywords that have frequencies higher than 10 for one day volume. 
        # Note that, for different day, the set of keywords will be different. 
        glb.glb_dict_keywords_ref = Keyword.calc_keywords_ref(day_raw_tweets, glb) # key is keyword, value is frequency of the keyword in raw tweets 
        glb.glb_dict_users = User.parse_users(day_raw_tweets, glb) # This function will add users mentioned in the tweets into the dictionary "dict_tweets"
        for key, user in glb.glb_dict_users.items():
            glb.dict_userid_2_scrname[user.screen_name] = user.id
            
        # Step 2: Parse Tweets. Update dict_tweets, dict_keywords, dict_links, dict_city_keywords, glb.glb_dict_hashtags
        glb = Tweet.parse_raw_tweets(day_raw_tweets, glb)
        glb.glb_dict_city_2_keywords = dict()
        for ct_latlong, kw_name in glb.glb_dict_city_keywords:
            try:
                glb.glb_dict_city_2_keywords[ct_latlong][kw_name] = glb.glb_dict_city_keywords[ct_latlong, kw_name]
            except:
                glb.glb_dict_city_2_keywords[ct_latlong] = dict()
                glb.glb_dict_city_2_keywords[ct_latlong][kw_name] = glb.glb_dict_city_keywords[ct_latlong, kw_name]
                
#        print 'start to generate keywords:'
#        sys.stdout.flush()
        # Step 3: Update country.state.city.dict_keywords
        for (co_name, country) in glb.glb_dict_countries.items():
            for (st_name, state) in country.dict_states.items():
                for (ci_latlong, city) in state.dict_cities.items():
                    try:
                        city.dict_keywords = glb.glb_dict_city_2_keywords[ci_latlong] # if the key does not exist, then that means no keywrods or tweets in that city
                    except:
                        city.dict_keywords = dict()
                    state.dict_user_ids.update(city.dict_user_ids)
                    state.dict_tweet_ids.update(city.dict_tweet_ids)
                state.update_dict_keywords_from_cities()
                for kw_name in state.dict_keywords:
                    glb.glb_dict_state_keywords[co_name, st_name, kw_name] = state.dict_keywords[kw_name]
                country.dict_user_ids.update(state.dict_user_ids)
                country.dict_tweet_ids.update(state.dict_tweet_ids)
            country.update_dict_keywords_from_states()
            for kw_name in country.dict_keywords:
                glb.glb_dict_country_keywords[co_name, kw_name] = country.dict_keywords[kw_name]
    
        ## need to update state and country level tweet ids and user ids
    
#        print 'start to generate mention graph: '
#        sys.stdout.flush()
        glb.glb_mention_graph = Mentions_graph()
        glb.glb_mention_graph.build_graph(glb)
        for index, subgraph_node_ids in enumerate(glb.glb_mention_graph.subgraphs):
            for user_id in subgraph_node_ids:
                user = glb.glb_dict_users[user_id]
                user.mention_group_id = index
    
#        print 'start to generate diffusion graph: '
#        sys.stdout.flush()
        glb.glb_diffusion_graph = Diffusion_graph()
        glb.glb_diffusion_graph.build_graph(glb)
        for index, subgraph in enumerate(glb.glb_diffusion_graph.subgraphs):
            for user_id in subgraph.keys():
                user = glb.glb_dict_users[user_id]
                user.diffusion_group_id = index
    
#        print 'start to generate follow graph: '
#        sys.stdout.flush()
        glb.glb_follow_graph = Follow_graph()
        glb.glb_follow_graph.build_graph(glb)
        for index, subgraph in enumerate(glb.glb_mention_graph.subgraphs):
            for user_id in subgraph:
                user = glb.glb_dict_users[user_id]
                user.follow_group_id = index
    
#        print 'start to generate retweet graph: '
        glb.glb_retweet_graph = Retweet_graph()
        glb.glb_retweet_graph.build_graph(glb)
        for index, subgraph in enumerate(glb.glb_retweet_graph.subgraphs):
            for tweet_id in subgraph:
                tweet = glb.glb_dict_tweets[tweet_id]
                tweet.retweet_group_id = index
    
#        print 'start to generate reply graph: '
#        sys.stdout.flush()
        glb.glb_reply_graph = Reply_graph()
        glb.glb_reply_graph.build_graph(glb)
        for index, subgraph in enumerate(glb.glb_reply_graph.subgraphs):
            for tweet_id in subgraph:
                tweet = glb.glb_dict_tweets[tweet_id]
                tweet.reply_group_id = index
#        print 'reply graph finished'
#        sys.stdout.flush()
#
#        print 'features calculated for all objects'        
#        sys.stdout.flush()
        glb.calc_features()
        
        return glb

    @staticmethod
    def roc_calc(scores):
        sort_obj_stat = array(sorted(scores, key = lambda x:x[0] * -1))
        precs = []
        recalls = []
        for i in range(len(sort_obj_stat)):
            recall = (sort_obj_stat[0:i,1].sum() * 1.0) / len(scores)
            precision = (sort_obj_stat[0:i,1].sum() * 1.0) / i
            recalls.append(recall)            
            precs.append(precision)
        return precs, recalls

    """
    scores is a list of size 2 lists
    [[score, 0 or 1], ...]
    """
    @staticmethod
    def threshold_calc(scores):
        sort_obj_stat = array(sorted(scores, key = lambda x:x[0] * -1))
#        print sort_obj_stat
        num_pos = sum(item[1] for item in scores)
        max_i = None
        max_f_measure = None
        max_score = None
        for i in range(len(sort_obj_stat)):
            recall = (sort_obj_stat[0:i+1,1].sum() * 1.0) / num_pos
            precision = (sort_obj_stat[0:i+1,1].sum() * 1.0) / (i + 1)
            f_measure = funs.f_measure(precision, recall)
#            print f_measure
            if max_f_measure == None or max_f_measure < f_measure:
                max_f_measure = f_measure
                max_i = i 
                max_score = sort_obj_stat[i,0]
        return max_score, f_measure
        
    """
    scores is a list of size 2 lists
    [[score, 0 or 1], ...]
    """
    @staticmethod
    def threshold_calc1(scores):
        sort_obj_stat = array(sorted(scores, key = lambda x:x[0] * -1))
#        print sort_obj_stat
        max_f_score = None
        max_score = None
        for i in range(len(sort_obj_stat)):
            n_pos = sort_obj_stat[0:i+1,1].sum()
            n_neg = (i+1) - n_pos
            f_score = n_pos - 30 * n_neg
#            print f_measure
            if max_f_score == None or max_f_score < f_score:
                max_f_score = f_score
                max_score = sort_obj_stat[i,0]
        return max_score, max_f_score

    @staticmethod
    def reformat_gsr_dict(all_gsr):
        co_date_dict = dict()
        for co in all_gsr:
            if co_date_dict.has_key(co) is not True:
                co_date_dict[co] = dict()
            for str_dt_time in all_gsr[co]:
                dt_time = funs.gt(str_dt_time)
                dt = dt_time.date()
                co_date_dict[co][dt] = all_gsr[co][str_dt_time]
        return co_date_dict

    @staticmethod    
    def cdf_calc(x):
        x = int(floor(x * 100))
        if x >= 400:
            return 1
        else:
            return glb_cdf_values[x]

    @staticmethod
    def f_measure(prec, recall):
        if prec + recall > 0:
            return (2 * prec * recall * 1.0)  / (prec + recall)
        else:
            return 0
        
    @staticmethod
    def get_features_info(obj):
        info = ''
        for feature, feature_title in zip(obj.features, obj.desp_features):
            info = info + ' {}:{}'.format(feature_title, feature)
        return info

    @staticmethod
    def update_loc_info(obj, geo_location):
        city = geo_location.city
        state = geo_location.state
        country = geo_location.country
        if city:
            obj.ci_latlong = city.latlong
            obj.ci_name = city.name
        if state:
            obj.st_name = state.name
        if country:
            obj.co_name = country.name
    
    @staticmethod
    def get_anomaly_desp(obj, glb, stat, unified_stat):
        anomaly_desp = ''
        for index, feature_pvalue in enumerate(obj.feat_pvalues):
            if feature_pvalue < glb.alpha:
                _, mean, std = Anomalies.select_mean_std(obj.features[index], index, stat, unified_stat)
                mean_str = '{:.3f}'.format(mean)
                std_str  = '{:.3f}'.format(std)
                feature_pvalue_str = '{:.3f}'.format(feature_pvalue)
                feature = '{:.3f}'.format(obj.features[index]) 
                anomaly_desp = anomaly_desp + ' {}: {}({} | {}, {})'.format(obj.desp_features[index], feature, feature_pvalue_str, mean_str, std_str)
        return anomaly_desp

    @staticmethod
    def get_basic_graph_info(obj):
        basic_info = ''
        for index, feature_pvalue in enumerate(obj.features):
            basic_info = basic_info + ' {}: {}({})'.format(obj.desp_features[index], obj.features[index], obj.feat_pvalues[index])
        return basic_info

    @staticmethod
    def concatenate_arrays(col1, col2):
#        print col1
        if len(col1.shape) == 1:
            n_cols = col1.shape[0]
            return concatenate((col1.reshape(n_cols,1), col2.reshape(n_cols,1)), axis = 1)
        else:
            return concatenate((col1, col2), axis = 1)
            
    @staticmethod
    def str_code(s):
        try:
            s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
        except TypeError:
            pass
        return s
        
    @staticmethod
    def str_len_dict(dict_obj):
        return str(len(dict_obj.keys()))
    
    @staticmethod
    def new_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    @staticmethod
    def gt(dt_str):
        try:
            dt_str = dt_str.replace('+00:00', '')
            dt, _, us = dt_str.partition(".")
            dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
        except: 
            print dt_str
            dt = None
        return dt
    
    @staticmethod
    def gt1(dt_str):
        try:
            dt_str = dt_str.replace('+00:00', '')
            dt, _, us = dt_str.partition(".")
            dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        except:
            print dt_str
            dt = None
        return dt

    @staticmethod
    def gt2(dt_str):
        try:
            dt = datetime.strptime(dt_str, TIME_FORMAT)
#            dt = datetime(*(time.strptime(dt_str, TIME_FORMAT)[0:6]))
        except:
            print dt_str
            dt = None
        return dt
 

    @staticmethod
    def save_to_pkl_file(obj, out_file):
    
        with open(out_file, "wb") as output:
            pickle.dump(obj, output,  pickle.HIGHEST_PROTOCOL)
#        with open(out_file, "wb") as output:
#            cPickle.dump(obj, output, cPickle.HIGHEST_PROTOCOL)
    
    ##    fp = open(out_file, 'wb')
    ##    pickle.dump(obj, fp)
    ##    fp.close()
    @staticmethod
    def load_pkl_file(in_file):
        f = open(in_file, "rb")
        obj = pickle.load(f) # protocol version is detected
        return obj

    @staticmethod
    def json_load(f):
        json_data = open(f, 'r')
        data = json.load(json_data, encoding='utf-8')
        return data
    
    
    @staticmethod
    def json_save(f, data):
        f = open(f, "w")
        f.write(json.dumps(data))
        f.close()
    
    @staticmethod
    def calc_geo_location(raw_tweet, glb):
        city = None
        country = None
        admin1 = None
#        try: 
#            latlong = raw_tweet['loc']['coordinates']
#            city, country, admin1 = glb.latlong_2_city[(latlong[0], latlong[1])]
##            print city, country, admin1glb_geo
#        except:
##            print raw_tweet
#            city, country, admin1, admin2, admin3 = glb_geo.geo_normalize(raw_tweet)
#            if city and admin1:
#                latlong = glb_geo.map_latlong[(city, country, admin1, admin2, admin3)]
    
        LT = glb_geo.feng_geo_normalize(raw_tweet)
        city, country, admin1, admin2, admin3, pop, lat, lon = LT[0:8]
        
        if city:
            city = city.lower()
        if country:
            country = country.lower()
            
        if admin1:
            admin1 = admin1.lower()
            
        latlong = (lat, lon)
        
#        if glb.latlong_2_ci_co_adms.has_key(latlong):
#            city, country, admin1, admin2, admin3 = glb.latlong_2_ci_co_adms[latlong] 
##            print city, country, admin1, admin2, admin3, pop, lat, lon
#        elif lat and lon:
##            print latlong, city, country, admin1
#            re = glb_geo.lookup_city(latlong[0], latlong[1])
##            print re
#            (city, country, admin1, admin2, admin3, pop, lat, lon, _, _) = re[0]
##            print '**********************************************'
##            print city, country, admin1, admin2, admin3, pop, lat, lon
#            latlong = (lat, lon)
#        if city and admin1:
#            latlong = glb_geo.map_latlong[(city, country, admin1, admin2, admin3)]

#        country = country.lower()
#        admin1 = admin1.lower()
#        city = city.lower()

        if country not in glb.glb_dict_countries.keys():
            print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
            print city, country, admin1
            return None
        
        if city is not None and glb.glb_dict_cities.has_key((latlong[0], latlong[1])):
#            print city, country, admin1
            geo_location = Geographic_location(glb.glb_dict_cities[(latlong[0], latlong[1])], glb.glb_dict_countries[country].dict_states[admin1], glb.glb_dict_countries[country])
        elif admin1 is not None and glb.glb_dict_countries[country].dict_states.has_key(admin1):
            geo_location = Geographic_location(None, glb.glb_dict_countries[country].dict_states[admin1], glb.glb_dict_countries[country])
        elif country  is not None:
            geo_location = Geographic_location(None, None, glb.glb_dict_countries[country])
        else:
            geo_location = None
            
        return geo_location
        
    @staticmethod
    def load_dict_countries_states_cities(countries, glb):
    
        dict_countries = dict()
        dict_cities = dict()
    
        for country_name in countries:
            country = Country(country_name)
            dict_countries[country_name] = country
    
        data = funs.load_pkl_file('flt_location_data.pkl')
        
        data = [(ci, co, st, a2, a3, p, lat, lon) for (ci, co, st, a2, a3, p, lat, lon) in data if co in countries]
    
#        print dict_countries['mexico'].dict_states
        for (ci, co, st, a2, a3, p, lat, lon) in data:
            try: # Neglect locations where the corresponding countries are not in the list
                if co:
                    co = co.lower()
                    country = dict_countries[co]
                    if ci:
                        city = City(ci, (lat,lon))
                        city.co_name = country.name
                        dict_cities[city.latlong] = city
                    if st:
                        if not country.dict_states.has_key(st):
                            country.dict_states[st] = State(st)
                        state = country.dict_states[st]
                        state.co_name = country.name
                        city.st_name = st
                        state.dict_cities[city.latlong] = city
#                    dict_city_keywords[city.latlong] = dict()
            except:
                pass

        for co_name, country in dict_countries.items():
#            print co_name
            for st_name, state in country.dict_states.items():
                print st_name
                state.update_latlong() # calcualte the center of the cities in the state as the state location
                state.build_city_neighborhood_graph()
                state.kdtree = None
            country.update_cities() # merge cities in states into a single dictionary
            country.build_state_neighborhood_graph() # built based on k nearest neighbors
            country.kdtree = None

        dict_states = dict()
        for co_name, country in dict_countries.items():
            for st_name, state in country.dict_states.items():
                dict_states[(co_name, st_name)] = state
    
        return dict_countries, dict_states, dict_cities


    @staticmethod
    def embers_stem(x):
        """
        DESCRIPTION
        It will do stemming for words in x considering english, spanish and portuguese
    
        INPUT
        x: a tweet text, or other sentense or paragraph
    
        OUTPUT
        the tweet text after stemming.
    
        """
        x = x.lower()
        if isinstance(x, unicode) == False:
            x = x.decode('utf-8', 'ignore')
        try:
            stemmer = SnowballStemmer('spanish')
            x1  = FeatureCountVectorizer.preprocess_unicode_text(x,stemmer.stem)
            if(x1 == ''):
                x1 = x
        
            stemmer = SnowballStemmer('english')
            x2  = FeatureCountVectorizer.preprocess_unicode_text(x,stemmer.stem)
            if(x2 == ''):
                x2 = x
        
            stemmer = SnowballStemmer('portuguese')
            x3  = FeatureCountVectorizer.preprocess_unicode_text(x,stemmer.stem)
            if(x3 == ''):
                x3 = x
    #        print 'success'
            return min(x1, x2, x3, key = lambda x: len(x))
        except:
            return x

def main():
    
    str = 'examples'
    print funs.embers_stem(str)
    pass
if __name__ == '__main__':
    main()

