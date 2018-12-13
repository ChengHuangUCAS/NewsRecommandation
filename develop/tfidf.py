
# coding: utf-8

# In[55]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import json
import codecs
import jieba
import numpy
import math


# In[119]:


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
    for news_key in news_dict:
        title_text = news_dict[news_key][0]
        _title = jieba.lcut(title_text)
        for w in _title[:]:
            if w.split('.')[0].isdigit():
                _title.remove(w)
        title = ' '.join(_title)
        title_array.append(title)
        
        doc_text = news_dict[news_key][1]
        _doc = jieba.lcut(doc_text)
        for w in _doc[:]:
            if w.split('.')[0].isdigit():
                _doc.remove(w)
        doc = ' '.join(_doc)
        doc_array.append(doc)
        
        i += 1
        if i % 1000 == 0:
            print(i)
    
    scale = max(int(title_scale / doc_scale), 1)
    doc_array = title_array * scale + doc_array
    
    # tf-idf算法，文章转化为一个归一化[并没有]的向量
    tfidf_vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df, stop_words = stop_list)
    tfidf_vectorizer.fit(doc_array)
    doc_matrix = tfidf_vectorizer.transform(doc_array)
    news_matrix = doc_matrix.todense().tolist()
    
#     word_bag = {}
#     for key in tfidf_vectorizer.vocabulary_:
#         word_bag.setdefault(tfidf_vectorizer.vocabulary_[key], key)
    
    # 构建news_key : vector字典
    i = 0
    news_vector_dict = {}
    for news_key in news_dict:
        news_vector_dict.setdefault(news_key, news_matrix[i])
        if i % 1000 == 0:
            print('i='+str(i))
            
         #打印文章关键词和权重
#         if i < 15:
#             news_words = []
#             news_words_weight = []
#             for j in range(len(news_matrix[i])):
#                 if news_matrix[i][j] > 0:
#                     news_words.append(word_bag[j])
#                     news_words_weight.append(news_matrix[i][j])
#             print(news_words)
#             print(title_array[i])
#             print(doc_array[i])
#             print(news_words_weight)

            print('-------------------------')
    
        i += 1

    return news_vector_dict

file_root = './data/_news_data_clean.json'
# NOTE: scale = MAX(int(title_scale / doc_scale), 1)
title_scale = 0.5
doc_scale = 1.0 - title_scale
min_df = 3
max_df = 25
_news_vector_dict = news_vector_dict(file_root, title_scale, doc_scale, min_df, max_df)


# In[135]:


def time_back(t):
    a = int(t-1393603200)
    return float(a / 86400 / 20)

def user_vector_dict(news_vector_dict):
    file = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
    news_data = codecs.open('./data/_news_data_clean.json', 'r', 'utf-8')
    user_dict = json.load(file)
    news_d = json.load(news_data)
    
    j = 0
    user_vector_dict = {}
    # 每一个用户
    for user_key in user_dict:
        # 该用户读过的所有新闻的向量和为用户向量
        i = 0
        vector_sum = numpy.matrix('0.0')
        for user_news_key in user_dict[user_key]:
            vector = numpy.matrix(news_vector_dict[user_news_key])
#             time_scale = time_back(news_d[user_news_key][2])
            time_scale = 1
            vector_sum = vector * time_scale + vector_sum
            i += 1
        if i != 0:
            vector_sum /= i
        user_vector_dict.setdefault(user_key, vector_sum.tolist()[0])
        j += 1
        if j % 1000 == 0:
            print('j='+str(j))
            print(vector_sum.tolist()[0][:10])
    return user_vector_dict


_user_vector_dict = user_vector_dict(_news_vector_dict)


# In[136]:


def k_n_n(news_dict, user_dict, k):
    news_keys = []
    news = []
    i = 0
    for news_key in news_dict:
        news_keys.append(news_key)
        news.append(news_dict[news_key])
        i += 1
        if i % 1000 == 0:
            print(i)
    
    print("training...")
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(news)
    
    user_keys = []
    nbrs = []
    i = 0
    for user_key in user_dict:
        users = []
        user_keys.append(user_key)
        users.append(user_dict[user_key])
        nbrs += neigh.kneighbors(users)
        i += 1
        if i % 50 == 0:
            print(i)
    
#     n = neigh.kneighbors(users)
    print(nbrs[:10])
    return nbrs

k = 200
n = k_n_n(_news_vector_dict, _user_vector_dict, k)


# In[137]:


# def time_scale(t):
#     day=int((t-1393603200)/86400)
#     if day < 1:
#         day = 1
#     return (math.log(day)+1)

# news_keys = []
# i = 0
# for news_key in _news_vector_dict:
#     news_keys.append(news_key)

    
# file = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
# f_news_data = codecs.open('./data/_news_data_clean.json', 'r', 'utf-8')
# user_news_dict = json.load(file)
# news_data = json.load(f_news_data)

# result = {}
# i = 0
# lens = []
# pair = []
# for user_key in user_news_dict:
#     dist = n[2*i][0].tolist()
#     indices = n[2*i+1][0].tolist()
#     pair = []
#     for m in range(len(indices)):
#         mth = indices[m]
#         news_id = news_keys[mth]
#         if news_id in user_news_dict[user_key]:
#             continue
    
#         time_ratio = time_scale(news_data[news_id][2])
#         #print(time_ratio, dist[m], dist[m]*time_ratio)
#         pair.append([dist[m] * time_ratio, news_id])
    
#     pair.sort(key=lambda x:x[0],reverse=True)
#     user_news_keys = []
#     for k in range(30):
#         user_news_keys.append(pair[k][1])
#     result.setdefault(user_key, user_news_keys)
#     if i < 10:
#         for p in pair:
#             print(p[0], time_scale(news_data[p[1]][2]))
#         print("\n")
    
# #         print(len(result[user_key]))
# #         print(result[user_key])
# #         print(user_news_dict[user_key])
#     i += 1
#     lens.append(len(result[user_key]))
# lens.sort()
# print(lens[:10])



news_keys = []
i = 0
for news_key in _news_vector_dict:
    news_keys.append(news_key)

    
file = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
user_news_dict = json.load(file)

result = {}
i = 0
lens = []
for user_key in user_news_dict:
    indices = n[2*i+1][0].tolist()
    user_news_keys = []
    for index in indices:
        user_news_key = news_keys[index]
        if user_news_key not in user_news_dict[user_key]:
            user_news_keys.append(user_news_key)
    result.setdefault(user_key, user_news_keys)
#     if i < 100:
#         print(len(result[user_key]))
#         print(result[user_key])
#         print(user_news_dict[user_key])
    i += 1
    lens.append(len(result[user_key]))
lens.sort()
print(lens[:10])



# In[138]:


file_output = codecs.open('./data/tfidf_result.json', 'w', 'utf-8')
json.dump(result, file_output)
file_output.close()


# In[139]:


def time_back(t):
    a = int(t-1393603200)
    return int(a / 86400)

def test(result_root):
    f_user_data_validation = codecs.open('./data/_user_data_validation_clean.json', 'r', 'utf-8')
    f_result = codecs.open(result_root, 'r', 'utf-8')
    f_news_data = codecs.open('./data/_news_data.json', 'r', 'utf-8')
    f_user_data_training = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
    training = json.load(f_user_data_training)
    validation = json.load(f_user_data_validation)
    result = json.load(f_result)
    news_data = json.load(f_news_data)
    
    z=0
    q=0
    user_num = 0
    precision = 0
    recall = 0
    sum=0
    for key in validation:
        user_num += 1
        if key in result:
            rec_num = len(result[key])
        else:
            continue
        act_num = len(validation[key])
        TP = 0
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
        sum+=TP
        for a in validation[key]:
            if time_back(news_data[a][2]) < 10:
                z+=1
#         if TP == 0:
#             print(key)
#             for a in result[key]:
#                 print(news_data[a][0], time_back(news_data[a][2]))
#             print("\n")
#             for a in validation[key]:
#                 print(news_data[a][0], time_back(news_data[a][2]),time_back(validation[key][a]))
#             print("\n")
#             for a in training[key]:
#                 print(news_data[a][0], time_back(news_data[a][2]),time_back(training[key][a]))
#             print("\n")
#             break
    
#         print(precision, recall)
        precision += TP / rec_num
        recall += TP / act_num
    precision = precision / user_num 
    recall = recall / user_num
    print(z,q,sum)
    f_user_data_validation.close()
    f_result.close()
    print("precision: ", precision )
    print("recall: ", recall)

test('./data/tfidf_result.json')


# In[141]:


tfidf_vectorizer = TfidfVectorizer()
help(tfidf_vectorizer)

