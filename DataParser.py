#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import argparse
import random

from time import time
from pyspark import SparkContext
from helpers import strip_non_alpha, to_lower_case, is_link  


# In[2]:


sc = SparkContext("local[*]", "Data Parsing")


# In[3]:


rdd_raw = sc.textFile("data.txt")


# In[6]:


rdd_raw.count()


# In[4]:


sparse_rdd = rdd_raw.map(lambda line:((int(line.split(',')[0][1])),line))\
                    .mapValues(lambda line:' '.join(line.split(',')[5:]))\
                    .mapValues(lambda sentence: to_lower_case(sentence))\
                    .mapValues(lambda sentence: sentence.split())\
                    .mapValues(lambda words: [is_link(word) for word in words])\
                    .mapValues(lambda words: [strip_non_alpha(word) for word in words])\
                    .mapValues(lambda words: [word for word in words if word])\
                    .mapValues(lambda words: ' '.join(words))\
                    .map(lambda pairs:(pairs[0]//4 ,pairs[1]))


# In[5]:


sparse_data = sparse_rdd.collect()


# In[6]:


random.shuffle(sparse_data)


# In[12]:


train = sparse_data[0:int(len(sparse_data)*0.9)]
test  = sparse_data[int(len(sparse_data)*0.9):]


# In[13]:


ids = []
f= open("parsedData.txt","w")
for idx,item in enumerate(sparse_data):
    try:
        f.write("%s\n" % str(item))
    except:
        ids.append(idx)
        print(idx)
f.close() 


# In[14]:


ids = []
f= open("train.txt","w")
for idx,item in enumerate(train):
    try:
        f.write("%s\n" % str(item))
    except:
        ids.append(idx)
        print(idx)
f.close() 


# In[15]:


ids = []
f= open("test.txt","w")
for idx,item in enumerate(test):
    try:
        f.write("%s\n" % str(item))
    except:
        ids.append(idx)
        print(idx)
f.close() 


# In[ ]:




