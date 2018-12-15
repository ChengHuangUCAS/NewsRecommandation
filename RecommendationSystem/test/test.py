# coding: utf-8
from __future__ import division
import json
import codecs

def test(result_root):
    datapath = 'data/'
    f_result = codecs.open(result_root, 'r', 'utf-8')
    f_news_data = codecs.open(datapath + '_news_data.json', 'r', 'utf-8')
    f_user_data_training = codecs.open(datapath + '_user_data_training_clean.json', 'r', 'utf-8')
    f_user_data_validation = codecs.open(datapath + '_user_data_validation_clean.json', 'r', 'utf-8')
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
    try :
        f1 += 2*(precision*recall)/(precision+recall)
    except:
        f1 = 0
    print "number of records in validation set: " + str(z)
    print "shot:                                " + str(q)
    print "hits:                                " + str(TP)
    f_user_data_validation.close()
    f_result.close()
    print "precision: " + str(precision)
    print "recall:    " + str(recall)
    print "f1:        " + str(f1)

if __name__ == '__main__':
    print '>>>>>>test nmf<<<<<<'
    test('data/nmf_pred_matrix.json')
    #print '>>>>>>test userbase<<<<<<'
    #test('../data/user_based_pred_matrix.json')
    print '>>>>>>test knn<<<<<<'
    test('data/knn_pred_matrix.json')

