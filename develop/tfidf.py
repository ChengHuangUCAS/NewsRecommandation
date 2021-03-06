
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import json
import codecs
import jieba
import numpy
import math


# In[ ]:


def news_vector_dict(file_root, title_scale, doc_scale, min_df, max_df):
    file = codecs.open(file_root, 'r', 'utf-8')
    news_dict = json.load(file)
    
    # 停用词表
    stop_file = codecs.open('./data/stop_words.txt', 'r', 'utf-8')
    stop_list = stop_file.read().split('\n')
    stop_file.close()
    
    # 分词，在词之间加空格，重新组成文章
    i = 0
    title_array = []
    doc_array = []
    scale = max(int(title_scale / doc_scale), 1)
    for news_id, news_info in news_dict.items():
        title_text = news_info[0]
        _title = jieba.lcut(title_text)
        for w in _title[:]:
            if w.split('.')[0].isdigit():
                _title.remove(w)
        title = ' '.join(_title)
        title_array.append(title)
        
        doc_text = news_info[1]
        _doc = jieba.lcut(doc_text)
        for w in _doc[:]:
            if w.split('.')[0].isdigit():
                _doc.remove(w)
        doc = ' '.join(_doc)
        doc = title * scale + ' ' + doc
        doc_array.append(doc)
        
        i += 1
        if i % 1000 == 0:
            print(i)
    
    
    # tf-idf算法，文章转化为一个词向量
    tfidf_vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df, stop_words = stop_list)
    tfidf_vectorizer.fit(doc_array)
    doc_matrix = tfidf_vectorizer.transform(doc_array)
    news_matrix = doc_matrix.todense().tolist()
    
    # 构建news_key : vector字典
    i = 0
    _news_vector_dict = {}
    for news_key, news_info in news_dict.items():
        _news_vector_dict.setdefault(news_key, [numpy.asarray(news_matrix[i]), news_info[2]]) 
        i += 1
    print("done.")
    return _news_vector_dict

file_root = './data/_news_data_clean.json'
# NOTE: scale = MAX(int(title_scale / doc_scale), 1)
title_scale = 0.5
doc_scale = 1.0 - title_scale
min_df = 10
max_df = 70
_news_vector_dict = news_vector_dict(file_root, title_scale, doc_scale, min_df, max_df)


# In[ ]:


def time_trans(t):
    day=int((t-1393603200)/86400)
    return day

def time_scale(day):
    if day < 0:
        day = 0
    if day > 20:
        day = 20
#     return math.log(day+1)+1
#     return math.log(math.log(day+1)+1)+1
    return 1+0.05*day

def user_vector_dict(_news_vector_dict):
    file = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
    user_dict = json.load(file)
    
    j = 0
    _user_vector_dict = {}
    # 每一个用户
    for user_key, user_info in user_dict.items():
        # 该用户读过的所有新闻的向量和为用户向量
        i = 0
        vector_sum = numpy.matrix('0.0')
        for user_news_key in user_info:
            vector = numpy.matrix(_news_vector_dict[user_news_key][0])
            vector_sum = vector + vector_sum
            t_scale = time_scale(time_trans(_news_vector_dict[user_news_key][1]))
            vector_sum = vector * t_scale + vector_sum
            i += 1
        if i != 0:
            vector_sum /= i
        #_user_vector_dict = {user_key: [user_vector, [all news_id read by this_user]]}
        _user_vector_dict.setdefault(user_key, [numpy.asarray(list(vector_sum)[0]),list(user_info.keys())])
    print("done")
    return _user_vector_dict


_user_vector_dict = user_vector_dict(_news_vector_dict)


# In[ ]:


def cal_dist(_news_vector_dict, _user_vector_dict):
    dist_result = {}
    i=0
    for user_id,user_v in _user_vector_dict.items():
        i+=1
        if i % 50 == 0:
            print(i)
        dist_list = []
        read_news = user_v[1]
        # calculate the Euclidean distance between each user and each news
        for news_id,news_v in _news_vector_dict.items():
            if news_id not in read_news:
                tmp = user_v[0]-news_v[0]
                dist= math.sqrt((tmp*tmp).sum())
                dist_list.append([news_id, dist, time_trans(news_v[1])])
                
        dist_result.setdefault(user_id, dist_list)
    print("done.")
    # dist_result = {user_id: for all news read by user->[news_id, distance, news_day]}
    return dist_result


dist_result = cal_dist(_news_vector_dict, _user_vector_dict)


# In[ ]:


def stress_date(day, dates):
    # day: news_time
    # dates: user click time
    if day in dates:
        return 8
    if day+1 in dates:
        return 5
    if day+2 in dates:
        return 3
    return time_scale(day)
    

def get_dates():
    f_user_data_validation = codecs.open('./data/_user_data_validation_clean.json', 'r', 'utf-8')
    validation = json.load(f_user_data_validation)
    user_dates = {}
    for user, news in validation.items():
        dates = []
        for t in news.values():
            day = time_trans(t)
            if day not in dates:
                dates.append(day)
        user_dates.setdefault(user, dates)
        
    return user_dates

def find_k_nbr_2(dist_result, dates, k):
    # if user click news in day X, emphasis the weight of news release on X and X-1, X-2
    result = {}
    for user_id, dist in dist_result.items():
        dist_list=[]
        user_dates = dates[user_id]
        num = len(user_dates)
        for record in dist:
            dist_list.append([record[0], record[1]/stress_date(record[2], user_dates)])
        dist_list.sort(key=lambda x:x[1],reverse=False)
        news = []
        for j in range(k*num):
            news.append(dist_list[j][0])
        result.setdefault(user_id, news)
    return result


# In[ ]:


k = 15
dates = get_dates()
result_2 = find_k_nbr_2(dist_result, dates, k)
file_output = codecs.open('./data/tfidf_result.json', 'w', 'utf-8')
json.dump(result_2, file_output)
file_output.close()


# In[ ]:


def test(result_root):
    f_result = codecs.open(result_root, 'r', 'utf-8')
    f_news_data = codecs.open('./data/_news_data.json', 'r', 'utf-8')
    f_user_data_training = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
    f_user_data_validation = codecs.open('./data/_user_data_validation_clean.json', 'r', 'utf-8')
    validation = json.load(f_user_data_validation)
    training = json.load(f_user_data_training)
    result = json.load(f_result)
    news_data = json.load(f_news_data)
    
    z=0
    q=0
    user_num = 0
    rec_num = 0
    act_num = 0
    precision = 0
    recall = 0
    f1 = 0
    TP = 0
    sum=0
    news = []
    for key, news_info in validation.items():
        for news_id in news_info:
            if news_id not in news:
                news.append(news_id)
    for key in validation:
        user_num += 1
        if key in result:
            rec_num += len(result[key])
        else:
            continue
        act_num += len(validation[key])
#         TP = 0 
        for news_id in result[key]:
            q+=1
    #         print(news_data[news_id][0])
            if news_id in validation[key]:
                TP+=1

#         print(key)
#         for a in validation[key]:
#             print(news_data[a][0])
#         print("\n")
#         for a in training[key]:
#             print(news_data[a][0])
#         print("\n")
#         sum+=TP
        for a in validation[key]:
            z+=1
    precision += TP / rec_num
    recall += TP / act_num
#     precision = precision / user_num 
#     recall = recall / user_num
    f1 += 2*(precision*recall)/(precision+recall)
    print("number of records in validation set:", z)
    print("shot:                               ", q)
    print("hits:                               ", TP)
    f_user_data_validation.close()
    f_result.close()
    print("precision: ", precision )
    print("recall:    ", recall)
    print("f1:        ", f1)

test('./data/tfidf_result.json')

