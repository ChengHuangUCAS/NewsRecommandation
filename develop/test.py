#!/usr/bin/env python
# coding: utf-8

# In[2]:


# coding: utf-8
import json
import codecs
import numpy


# In[16]:


def test(result_root):
    f_user_data_validation = codecs.open('./data/_user_data_validation_clean.json', 'r', 'utf-8')
    f_result = codecs.open(result_root, 'r', 'utf-8')
    f_news_data = codecs.open('./data/_news_data.json', 'r', 'utf-8')
    f_user_data_training = codecs.open('./data/_user_data_training_clean.json', 'r', 'utf-8')
    training = json.load(f_user_data_training)
    validation = json.load(f_user_data_validation)
    result = json.load(f_result)
    news_data = json.load(f_news_data)
    
    
    precision = 0
    recall = 0
    for key in validation:
        if key in result:
            rec_num = len(result[key])
        else:
            continue
        act_num = len(validation[key])
        TP = 0
        for news_id in result[key]:
    #         print(news_data[news_id][0])
            if news_id in validation[key]:
                TP+=1

    #     print(key)
    #     print("\n")
    #     for a in validation[key]:
    #         print(news_data[a][0])
    #     print("\n")
    #     for a in training[key]:
    #         print(news_data[a][0])
    #     print("\n")
    #     if TP!=0:
    #         print(key)
    #         for a in result[key]:
    #             print(news_data[a][0])
    #         print("\n")
    #         for a in validation[key]:
    #             print(news_data[a][0])
    #         print("\n")
    #         for a in training[key]:
    #             print(news_data[a][0])
    #         print("\n")
    #     break
        precision += TP / rec_num
        recall += TP / act_num

    f_user_data_validation.close()
    f_result.close()
    print("precision: ", precision )
    print("recall: ", recall)


# In[17]:





# In[18]:


test('./data/CF_NMF_recommend_matrix.json')


# In[ ]:




