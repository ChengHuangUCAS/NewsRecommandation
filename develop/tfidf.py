
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import json
import codecs
import jieba
import numpy


# In[2]:


def news_vector_dict(title_scale, doc_scale):
    file = codecs.open('./data/_news_data.json', 'r', 'utf-8')
    news_dict = json.load(file)
    
    # 分词，在词之间加空格，重新组成文章
    i = 0
    title_array = []
    doc_array = []
    for news_key in news_dict:
        title_text = news_dict[news_key][0]
        doc_text = news_dict[news_key][1]
        title = ' '.join(jieba.lcut(title_text))
        doc = ' '.join(jieba.lcut(doc_text))
        title_array.append(title)
        doc_array.append(doc)
        i += 1
        if i % 1000 == 0:
            print(i)
    
    # tf-idf算法，文章转化为一个归一化的向量
    tfidf_vectorizer = TfidfVectorizer(min_df = 10)
    doc_matrix = tfidf_vectorizer.fit_transform(doc_array)
    title_matrix = tfidf_vectorizer.transform(title_array)
    
    # 计算文章加权vector
    news_matrix = (title_matrix.todense() * title_scale + doc_matrix.todense() * doc_scale).tolist()
    
    # 构建news_key : vector字典
    i = 0
    news_vector_dict = {}
    for news_key in news_dict:
#         news_vector = title_matrix[i].todense()
        news_vector_dict.setdefault(news_key, news_matrix[i])
        i += 1
        if i % 1000 == 0:
            print('i='+str(i))
            print(news_matrix[i][:10])
    # file_output = codecs.open('./data/_news_data_tfidf.json', 'w', 'utf-8')
    # json.dump(news_vector_dict, file_output)
    # print(tfidf_vectorizer.vocabulary_) 
    
    return news_vector_dict


# In[3]:


def user_vector_dict(news_vector_dict, time_scalse):
    file = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
    user_dict = json.load(file)
    
    j = 0
    user_vector_dict = {}
    # 每一个用户
    for user_key in user_dict:
        # 该用户读过的所有新闻的向量和为用户向量
        i = 0
        vector_sum = numpy.matrix('0.0')
        for user_news_key in user_dict[user_key]:
            vector = numpy.matrix(news_vector_dict[user_news_key])
            vector_sum = vector * time_scale + vector_sum
            i += time_scale
        if i != 0:
            vector_sum /= i
        user_vector_dict.setdefault(user_key, vector_sum.tolist()[0])
        j += 1
        if j % 1000 == 0:
            print('j='+str(j))
            print(vector_sum.tolist()[0][:10])
    return user_vector_dict


# In[4]:


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


# In[5]:


title_scale = 0.5
doc_scale = 1.0 - title_scale
time_scale = 1.0
k = 30
news_vector_dict = news_vector_dict(title_scale, doc_scale)


# In[6]:


user_vector_dict = user_vector_dict(news_vector_dict, time_scale)


# In[7]:


n = k_n_n(news_vector_dict, user_vector_dict, k)


# In[8]:


print(1)
print(n[2][0])


# In[9]:


news_keys = []
i = 0
for news_key in news_vector_dict:
    news_keys.append(news_key)

result = {}
i = 0
for user_key in user_vector_dict:
    indices = n[2*i+1][0].tolist()
    user_news_keys = []
    for index in indices:
        user_news_keys.append(news_keys[index])
    result.setdefault(user_key, user_news_keys)
    if i < 10:
        print(result[user_key])
    i += 1


# In[13]:


file_output = codecs.open('./data/tfidf_result.json', 'w', 'utf-8')
json.dump(result, file_output)
file_output.close()

