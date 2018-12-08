
# coding: utf-8

# In[1]:


import jieba
import jieba.analyse as analyse
import json
import codecs


# In[2]:


file = codecs.open('_news_data.json', 'r', 'utf-8')
file_output = codecs.open('_news_data_tfidf.json', 'w', 'utf-8')
jsonArray = json.load(file)


# In[3]:


i = 0
for jsonKey in jsonArray :
    title = analyse.extract_tags(jsonArray[jsonKey][0], topK=7)
    for word in title[:]:
        if word.isdigit():
            title.remove(word)
    jsonArray[jsonKey][0] = title
    tags = analyse.extract_tags(jsonArray[jsonKey][1], topK=7)
    for word in tags[:]:
        if word.isdigit():
            tags.remove(word)
    jsonArray[jsonKey][1] = tags
    i = i + 1
    if i % 1000 == 0:
        print(i)
    if i > 10:
        break
    print(jsonArray[jsonKey])


# In[ ]:


json.dump(jsonArray, file_output)

