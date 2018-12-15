# coding: utf-8

import io
import json
import codecs
import numpy as np
from sklearn.decomposition import NMF

class CF_NMF:
    user_dict = {}
    news_dict = {}
    user_list = []
    news_list = []
    user_i = []
    news_i = []


    def __init__(self):
        pass


    def build_matrix(self):
        # load data
        # user_dict{,user_id:{news_id:click_time},}
        # news_dict{,news_id:[title,article,time],}
        fuser = '../data/_user_data_training.json'
        fnews = '../data/_news_data.json'
        with io.open(fuser,'r',encoding='utf-8') as fp1:
            self.user_dict = json.load(fp1)
        with io.open(fnews,'r',encoding='utf-8') as fp2: 
            self.news_dict = json.load(fp2)
        
        #for key,val in self.news_dict.items(): 
        #    print(val[1])
        #    print(type(val[1]))
        #    break

        self.user_list = [key for key in self.user_dict] 
        self.news_list = [key for key in self.news_dict]
        self.user_i = dict(zip(self.user_list, range(len(self.user_dict))))
        self.news_i = dict(zip(self.news_list, range(len(self.news_dict))))
    
        #build user-news matrix
        M_list = [[0.0]*len(self.news_dict) for i in range(len(self.user_dict))]
        for key,val in self.user_dict.items():
            for news in val:
                M_list[self.user_i[key]][self.news_i[news]] = 1.0
        return np.array(M_list)     
    

    def run_NMF(self, M):
        # run NMF on user-item matrix 
        model = NMF(n_components=10, init='random', random_state=0)
        U = model.fit_transform(M)
        I = model.components_
        # recomstruct user-item matrix
        UI_M = np.dot(U, I)
    
        # delete news that has been read
        filter = M < 1.0
        recommend_M  = UI_M * filter
        return recommend_M

    
    def mwrite(self, M, fname):
        # {user_id:[news_id]}
        user_news_list = []
        for user_news in M:
            # get top10 indexes of the elements in descending order
            # print user_news.sort()
            sorted_idx = user_news.argsort()[::-1][:9]
            top_news = [self.news_list[idx] for idx in sorted_idx]
            user_news_list.append(top_news)
        user_news_dict = dict(zip(self.user_list, user_news_list))
        
        with codecs.open(fname, 'w', 'utf-8') as fp:
            json.dump(user_news_dict, fp) 


if __name__ == '__main__':
    cf_nmf = CF_NMF()
    M = cf_nmf.build_matrix()
    print("build matrix finished")
    recommend_M = cf_nmf.run_NMF(M)
    print("run NMF finished")
    fname = '../data/CF_NMF_recommend_matrix.json'
    print("writing file to ", fname)
    cf_nmf.mwrite(recommend_M, fname)
