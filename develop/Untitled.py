#!/usr/bin/env python
# coding: utf-8

# In[43]:


# -*- coding: utf-8 -*-
import numpy
import codecs
import json
import time
import sys
import operator as op

fp = sys.stdout


# In[70]:


def preprocess():
    f = codecs.open('./data/user_click_data.txt', 'r', 'utf-8')
#     f_training = codecs.open('./data/_training.json', 'w', 'utf-8')
#     f_validation = codecs.open('./data/_validation.json', 'w', 'utf-8')
    f_user_data_training = codecs.open('./data/_user_data_training.json', 'w', 'utf-8')
    f_user_data_validation = codecs.open('./data/_user_data_validation.json', 'w', 'utf-8')
    f_news_data = codecs.open('./data/_news_data.json', 'w', 'utf-8')
    f_user_data_training_clean = codecs.open('./data/_user_data_training_clean.json', 'w', 'utf-8')
    f_user_data_validation_clean = codecs.open('./data/_user_data_validation_clean.json', 'w', 'utf-8')
    f_news_data_clean = codecs.open('./data/_news_data_clean.json','w','utf-8')
    i=0
    user_data_training = {}
    news_data = {}
    user_data_validation = {}
    user_data_training_clean = {}
    user_data_validation_clean = {}
    news_data_clean={}
    #news_data_validation = {}
    print("preprocessing starts...")
    for line in f:
        # progress bar
        i+=1
        if(i % 10000 == 20):
            fp.write('\r')
            fp.write("processing")
            for z in range(int(i / 10000)%5):
                fp.write(".")
            fp.write("        ")
        partitions = line.split('\t')
        # user_id, news_id, click_time, title, article, news_time
        user_id = int(partitions[0])
        news_id = int(partitions[1])
        click_time = int(partitions[2])
        # delete all dirty record
        if partitions[4] == 'NULL' or partitions[3] == '404':
            continue
        # deal with news_time
        try:
            tstp = transform_time(partitions[5])  
        except:
            tstp = int(1393603200)
        # data = {"user_id": user_id, "news_id": news_id, "click_time": click_time,
        #        "title": partitions[3], "article": partitions[4], "news_time": tstp}
        day = int((int(partitions[2]) - 1393603200) / 86400) + 1
        if day <= 20:
            # the first 20 days records belong to training set
#             json.dump(data, f_training)
            if user_id not in user_data_training:
                user_data_training.setdefault(user_id, {})
            user_data_training[user_id].setdefault(news_id, click_time)
        else:
            # the last 10 days records belong to validation set
#             json.dump(data, f_validation)
            if user_id not in user_data_validation:
                user_data_validation.setdefault(user_id, {})
            user_data_validation[user_id].setdefault(news_id, click_time)
            
        if news_id in news_data:
            if len(partitions[4]) > len(news_data[news_id][1]):
                # if necessary,
                # update the news info
                news_data[news_id][1] = partitions[4]
                news_data[news_id][2] = tstp
#             if op.eq(partitions[4], news_data[news_id][1]) != 0 and \
#             op.eq(partitions[3], news_data[news_id][0]) != 0 and news_data[news_id][2] != tstp:
        else:
            # [news_title, news_article, news_time]
            news_data.setdefault(news_id, [partitions[3], partitions[4], tstp])
    
    fp.write('\nphase 1 done.\n')
    user_validation = user_data_validation.keys()
    user_training = user_data_training.keys()
#     zombie = list(set(user_validation) ^ set(user_training))
#     ret1 = [ i for i in user_validation if i not in user_training ]
#     ret2 = [ i for i in user_training if i not in user_validation ]
#     user_list = [ i for i in user_training if i in user_validation ]
#     print(len(zombie), len(ret1), len(ret2), len(user_validation), len(user_training), len(user_list))
    # we only concern those who read news in both periods
    user_list = [ i for i in user_training if i in user_validation ]
    news_list = []
    z = 0
    for i in user_list:
        # those who read little news in one period are also not considered
        if len(user_data_validation[i]) >= 3 and len(user_data_training[i]) >= 3:
            z+=1
            user_data_validation_clean.setdefault(i, user_data_validation[i])
            user_data_training_clean.setdefault(i, user_data_training[i])
            for n in user_data_validation[i]:
                if n not in news_list:
                    news_list.append(n)
            for n in user_data_training[i]:
                if n not in news_list:
                    news_list.append(n)
    for n in news_list:
        news_data_clean.setdefault(n, news_data[n])
#     print(len(user_list)
#     print(z)
    print("recleaning phase done.")
    print("\nstatistics:")
    print("\ttotal number of users:                     10000")
    print("\tusers read in the first 20 days:          ", len(user_data_training))
    print("\tusers read in the last 10 days:           ", len(user_data_validation))
    print("\tusers read in both two periods:           ", len(user_list))
    print("\tusers read at least 3 news in each period:", z)
    print("\tnumber of valid news:                     ", len(news_data))
    print("\tnumber of news read by valid users:       ", len(news_data_clean))
    
    # dump into files
    json.dump(user_data_training_clean, f_user_data_training_clean)
    json.dump(user_data_validation_clean, f_user_data_validation_clean)
    json.dump(user_data_training, f_user_data_training)
    json.dump(user_data_validation, f_user_data_validation)
    json.dump(news_data, f_news_data)
    json.dump(news_data_clean, f_news_data_clean)
    # close files
#     f_training.close()
#     f_validation.close()
    f_user_data_training.close()
    f_user_data_validation.close()
    f_news_data.close()
    f_user_data_training_clean.close()
    f_user_data_validation_clean.close()
    f_news_data_clean.close()
    
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

# In[ ]:


file = open('./data/_news_data.json','r',encoding='utf-8')
s = json.load(file)
p=0
print(len(s))
for a in s:
    if s[a][2] == 1393603200:
        p+=1
print(p)

