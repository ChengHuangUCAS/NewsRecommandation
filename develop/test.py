#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8
import json
import codecs
import numpy


# In[2]:


def time_back(t):
    a = int(t-1393603200)
    return int(a / 86400)


# In[39]:


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


# In[ ]:





# In[40]:


test('./data/tfidf_result.json')


# In[ ]:





# In[ ]:




