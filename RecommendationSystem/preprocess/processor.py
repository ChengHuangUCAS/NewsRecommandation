# coding: utf-8
from __future__ import division
import numpy
import codecs
import json
import sys
import operator as op

from tools.utils import transform_time


def data_preprocess(k = 5, p = 5):
    datapath = 'data/'
    fp = sys.stdout
    f = codecs.open(datapath + 'user_click_data.txt', 'r', 'utf-8')
    f_user_data_training = codecs.open(datapath + '_user_data_training.json', 'w', 'utf-8')
    f_user_data_validation = codecs.open(datapath + '_user_data_validation.json', 'w', 'utf-8')
    f_news_data = codecs.open(datapath + '_news_data.json', 'w', 'utf-8')
    f_user_data_training_clean = codecs.open(datapath + '_user_data_training_clean.json', 'w', 'utf-8')
    f_user_data_validation_clean = codecs.open(datapath + '_user_data_validation_clean.json', 'w', 'utf-8')
    f_news_data_clean = codecs.open(datapath + '_news_data_clean.json','w','utf-8')
    user_data_training = {}
    news_data = {}
    user_data_validation = {}
    user_data_training_clean = {}
    user_data_validation_clean = {}
    news_data_clean={}
    
    i = 0
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
        # data = {"user_id": user_id, "news_id": news_id, "click_time": click_time,
        #        "title": partitions[3], "article": partitions[4], "news_time": tstp}
        user_id = int(partitions[0])
        news_id = int(partitions[1])
        click_time = int(partitions[2])
        # delete all dirty record
        if partitions[4] == 'NULL' or partitions[3] == '404':
            continue
        try:
            tstp = transform_time(partitions[5]) 
        except:
            continue
        #      tstp = int(1393603200)
        day = int((int(partitions[2]) - 1393603200) / 86400) + 1
        if day <= 20:
            # the first 20 days records belong to training set
            if user_id not in user_data_training:
                user_data_training.setdefault(user_id, {})
            user_data_training[user_id].setdefault(news_id, click_time)
        else:
            # the last 10 days records belong to validation set
            if user_id not in user_data_validation:
                user_data_validation.setdefault(user_id, {})
            user_data_validation[user_id].setdefault(news_id, click_time)
        
        if news_id in news_data:
            if len(partitions[4]) > len(news_data[news_id][1]):
                # if necessary,
                # update the news info
                # actually, this code doesn't work
                news_data[news_id][1] = partitions[4]
                news_data[news_id][2] = tstp
        else:
            # [news_title, news_article, news_time]
            news_data.setdefault(news_id, [partitions[3], partitions[4], tstp])
    fp.write('\nphase 1 done.\n')
    user_validation = user_data_validation.keys()
    user_training = user_data_training.keys()
    # we only concern those who read news in both periods
    user_list = [ i for i in user_training if i in user_validation ]
    news_list = []
    valid_user = 0
    valid_record_training = 0
    valid_record_validation = 0
    for i in user_list:
        # those who read little news in one period are also not considered
        if len(user_data_validation[i]) >= k and len(user_data_training[i]) >= p:
            valid_user+=1
            user_data_validation_clean.setdefault(i, user_data_validation[i])
            user_data_training_clean.setdefault(i, user_data_training[i])
            for n in user_data_validation[i]:
                valid_record_validation+=1
                if n not in news_list:
                    news_list.append(n)
            for n in user_data_training[i]:
                valid_record_training+=1
                if n not in news_list:
                    news_list.append(n)
    for n in news_list:
        news_data_clean.setdefault(n, news_data[n])
            
    print "recleaning phase done."
    print "\nstatistics:"
    print "\ttotal number of users:                           10000"
    print "\tusers read in the first 20 days:                 " + str(len(user_data_training))
    print "\tusers read in the last 10 days:                  " + str(len(user_data_validation))
    print "\tusers read in both two periods:                  " + str(len(user_list))
    print "\tusers read at least " + str(k) + " news in the first period:  " + str(valid_user)
    print "\tnumber of valid news:                            " + str(len(news_data))
    print "\tvalid record in the first 20 days:               " + str(valid_record_training)
    print "\tvalid record in the last 10 days:                " + str(valid_record_validation)
    print "\tnumber of news read by valid users:              " + str(len(news_data_clean))
    
    
    # dump into files
    json.dump(user_data_training_clean, f_user_data_training_clean)
    json.dump(user_data_validation_clean, f_user_data_validation_clean)
    json.dump(user_data_training, f_user_data_training)
    json.dump(user_data_validation, f_user_data_validation)
    json.dump(news_data, f_news_data)
    json.dump(news_data_clean, f_news_data_clean)
    # close files
    f_user_data_training.close()
    f_user_data_validation.close()
    f_news_data.close()
    f_user_data_training_clean.close()
    f_user_data_validation_clean.close()
    f_news_data_clean.close()
    f.close()

if __name__ == '__main__':
    data_preprocess(5, 1)
