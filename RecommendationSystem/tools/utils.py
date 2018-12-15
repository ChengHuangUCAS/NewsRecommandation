#coding=utf8
from __future__ import division
import time
import json
import codecs

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

def transform_time(t):
    # t :
    # 2014年03月xx日xx:xx
    tmp = t.split(u"年")
    year = str(tmp[0])
    tmp = tmp[1].split(u"月")
    mon = str(tmp[0])
    tmp = tmp[1].split(u"日")
    day = str(tmp[0])
    tmp = tmp[1].split(":")
    hour = str(tmp[0])
    minute = str(tmp[1].split("\r")[0])
    # 1393603200 >>> 2014-03-01 00:00:00
    a = year+'-'+mon+'-'+day+' '+hour+':'+minute+':'+'00'
    timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return int(timeStamp)

def get_dates():
    datapath = 'data/'
    f_user_data_validation = codecs.open(datapath + '_user_data_validation_clean.json', 'r', 'utf-8')
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
