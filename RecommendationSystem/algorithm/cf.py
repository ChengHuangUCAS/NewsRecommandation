# coding: utf-8
from __future__ import division
import json
import codecs
import math
import random
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

from tools.utils import get_dates

class cf_algorithm:
 
    def __init__(self):
        self.user_dict = {}
        self.news_dict = {}
        self.user_list = []
        self.news_list = []
        self.user_i = []
        self.news_i = []
        self.datapath = 'data/'

    def build_matrix(self):
        # load data
        # user_dict{,user_id:{news_id:click_time},}
        # news_dict{,news_id:[title,article,time],}
        fuser = self.datapath + '_user_data_training_clean.json'
        fnews = self.datapath + '_news_data_clean.json'
        with codecs.open(fuser, 'r', 'utf-8') as fp1:
            self.user_dict = json.load(fp1)
        with codecs.open(fnews, 'r', 'utf-8') as fp2: 
            self.news_dict = json.load(fp2)
        self.user_list = [key for key in self.user_dict] 
        self.news_list = [key for key in self.news_dict]
        self.user_i = dict(zip(self.user_list, range(len(self.user_dict))))
        self.news_i = dict(zip(self.news_list, range(len(self.news_dict))))
    
        # build user-news matrix
        M_list = [[0.0]*len(self.news_dict) for i in range(len(self.user_dict))]
        for key,val in self.user_dict.items():
            for news_id, clk_time in val.items():
                pub_day = (int(self.news_dict[news_id][2]) - 1393603200) / 86400.0 
                #clk_day = (int(clk_time) - 1393603200) / 86400.0 
                T = np.e**(pub_day / 10.0)
                #T = np.log(clk_day * pub_day)
                M_list[self.user_i[key]][self.news_i[news_id]] = 1.0 * T
        
        return normalize(np.array(M_list))
    

    def pred_NMF(self, M):
        # run NMF on user-item matrix 
        model_nmf = NMF(n_components=8, init='random', random_state=0)
        U = model_nmf.fit_transform(M)
        I = model_nmf.components_
        # recomstruct user-item matrix
        UI_M = np.dot(U, I)
    
        # delete news that has been read
        filter = M == 0.0
        pred_M  = UI_M * filter
        return pred_M


    #def pred_userbased(self, M):
    #    cos_sim = 1-pairwise_distances(M, metric="cosine")
    #    mean_user_rating = M.mean(axis=1)[:, np.newaxis]

    #    rating_diff = (M - mean_user_rating)
    #    pred_M = mean_user_rating + cos_sim.dot(rating_diff) / np.array([np.abs(cos_sim).sum(axis=1)]).T
    #    return pred_M


    #def pred_Itembased(self, M):
    #    pass


    def mwrite(self, M, fname):
        # {user_id:[news_id]}
        user_news_list = []
        for i in range(len(M)):
            # get top10 indexes of the elements in descending order
            user_news = M[i]
            num_dates = len(get_dates()[self.user_list[i]])
            if num_dates > 10:
                num_dates = 10
            sorted_idx = user_news.argsort()[::-1][:20*num_dates]
            #print num_dates
            sorted_idx = random.sample(sorted_idx, 10*num_dates)
            top_news = [self.news_list[idx] for idx in sorted_idx]
            user_news_list.append(top_news)
        user_news_dict = dict(zip(self.user_list, user_news_list))
        
        with codecs.open(fname, 'w', 'utf-8') as fp:
            json.dump(user_news_dict, fp) 


    def run(self, type):
        M = self.build_matrix()
        print "build matrix finished"
        fname = ''
        pred_M = []
        if type == 'nmf':
            pred_M = self.pred_NMF(M)
            print "run NMF finished" 
            fname = self.datapath + 'nmf_pred_matrix.json'
        elif type == 'user':
            pred_M = self.pred_userbased(M)
            print "run user based cf finished" 
            fname = self.datapath + 'user_based_pred_matrix.json'

        print "writing file to " + fname
        self.mwrite(pred_M, fname)
