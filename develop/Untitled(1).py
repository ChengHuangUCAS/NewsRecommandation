#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy
import codecs
import json
import time


# In[3]:


def preprocess():
    f = codecs.open('./user_click_data.txt', 'r', 'utf-8')
    f_training = codecs.open('./_training.json', 'w', 'utf-8')
    f_validation = codecs.open('./_validation.json', 'w', 'utf-8')
    f_user_data_training = codecs.open('./_user_data_training.json', 'w', 'utf-8')
    f_user_data_validation = codecs.open('./_user_data_validation.json', 'w', 'utf-8')
    f_news_data = codecs.open('./_news_data.json', 'w', 'utf-8')
    
    i=0
    user_data_training = {}
    news_data = {}
    user_data_validation = {}
    #news_data_validation = {}
    for line in f:
        i+=1
        if(i % 10000 == 0):
            print(i)
        partitions = line.split('\t')
        #user_id, news_id, click_time, title, article, news_time
        user_id = int(partitions[0])
        news_id = int(partitions[1])
        click_time = int(partitions[2])
        # deal with news_time
        try:
            tstp = transform_time(partitions[5])  
        except:
            tstp = int(1393603200)
        #data = {"user_id": user_id, "news_id": news_id, "click_time": click_time,
        #       "title": partitions[3], "article": partitions[4], "news_time": tstp}
        day = int((int(partitions[2]) - 1393603200) / 86400) + 1
        
        if day <= 20:
            # the first 20 days records belong to training set
            #json.dump(data, f_training)
            if user_id not in user_data_training:
                user_data_training.setdefault(user_id, {})
            user_data_training[user_id].setdefault(news_id, click_time)
        else:
            # the last 10 days records belong to validation set
            #json.dump(data, f_validation)
            if user_id not in user_data_validation:
                user_data_validation.setdefault(user_id, {})
            user_data_validation[user_id].setdefault(news_id, click_time)
            
        if news_id in news_data:
            if len(partitions[4]) > len(news_data[news_id][1]):
                # if necessary,
                # update the news info
                news_data[news_id][1] = partitions[4]
                news_data[news_id][2] = tstp
        else:
            #[news_title, news_article, news_time]
            news_data.setdefault(news_id, [partitions[3], partitions[4], tstp])
    
    # dump into files
    json.dump(user_data_training, f_user_data_training)
    json.dump(user_data_validation, f_user_data_validation)
    json.dump(news_data, f_news_data)
    # close files
    f_training.close()
    f_validation.close()
    f_user_data_training.close()
    f_user_data_validation.close()
    f_news_data.close()
    f.close()
    
def transform_time(t):
    # t :
    # 2014年03月xx日xx:xx
    tmp = t.split("月")
    tmp = tmp[1].split("日")
    day = int(tmp[0])
    tmp = tmp[1].split(":")
    hour = int(tmp[0])
    minute = int(tmp[1].split("\r")[0])
    # 1393603200 >>> 2014-03-01 00:00:00
    timestamp = 1393603200 + (day - 1) * 86400 + hour * 3600 + minute * 60
    return int(timestamp)

preprocess()


# In[ ]:


a = "2014-03-01 00:00:00"
timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
timeStamp = int(time.mktime(timeArray))
print(timeStamp)


#     2014-03-01 00:00:00
#     ↓
#     1393603200

# In[2]:


file = open('./_user_data_training.json','r',encoding='utf-8')
s = json.load(file)
print(s)


# In[ ]:




