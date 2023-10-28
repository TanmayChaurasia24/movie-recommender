#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credit.head()


# In[5]:


movie_dataset = movies.merge(credit, on='title')


# In[6]:


movie_dataset.head(1)


# In[7]:


#generes
#id
#keyword
#overview
#title
#cast
#crew

movie_dataset = movie_dataset[['genres', 'id', 'keywords', 'overview', 'title', 'cast', 'crew']]


# In[8]:


movie_dataset.info()


# In[9]:


movie_dataset.head(1)


# In[10]:


movie_dataset.isnull().sum()


# In[11]:


movie_dataset.dropna(inplace=True)


# In[12]:


movie_dataset.isnull().sum()


# In[13]:


movie_dataset.shape


# In[14]:


movie_dataset.duplicated().sum()


# In[15]:


movie_dataset.iloc[0].cast


# In[16]:


movie_dataset.iloc[0].genres


# In[36]:


import ast


# In[37]:


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[38]:


movie_dataset['genres'] = movie_dataset['genres'].apply(convert)


# In[39]:


movie_dataset.head()


# In[30]:


movie_dataset['keywords'] = movie_dataset['keywords'].apply(convert)
movie_dataset.head()


# In[22]:


def extract_dic(obj):
    m = []
    
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            m.append(i['name'])
    return m


# In[23]:


movie_dataset['crew'] = movie_dataset['crew'].apply(extract_dic)


# In[24]:


movie_dataset.head()


# In[42]:


def convert3(text):
    L=[]
    counter = 0
    for i in ast.literal_eval(text):
        if(counter < 3):
            L.append(i['name'])
        counter += 1
    return L


# In[46]:


movie_dataset.head(1)


# In[48]:


movie_dataset['overview'] = movie_dataset['overview'].apply(lambda x:x.split())


# In[49]:


movie_dataset.head(1)


# In[52]:


movie_dataset['tags'] = movie_dataset['overview']+movie_dataset['genres']+movie_dataset['keywords']+movie_dataset['cast']+movie_dataset['crew']


# In[53]:


movie_dataset.head()


# In[54]:


new = movie_dataset.drop(columns=['genres','keywords','cast','crew','overview'])


# In[55]:


new.head(1)


# In[57]:


new['tags'] = new['tags'].apply(lambda x:" ".join(x))


# In[58]:


new.head()


# In[59]:


new['tags'] = new['tags'].apply(lambda x:x.lower())


# In[60]:


new['tags'][0]


# In[69]:


import nltk


# In[70]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[71]:


def stemming(text):
    list = []
    
    for i in text.split():
        list.append(ps.stem(i))
    return " ".join(list)


# In[72]:


new['tags'] = new['tags'].apply(stemming)


# In[73]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000 , stop_words = 'english')


# In[74]:


vector = cv.fit_transform(new['tags']).toarray()


# In[75]:


vector


# In[76]:


vector = cv.fit_transform(new['tags']).toarray().shape


# In[77]:


vector = cv.fit_transform(new['tags']).toarray()


# In[78]:


cv.fit_transform(new['tags']).toarray().shape


# In[79]:


from sklearn.metrics.pairwise import cosine_similarity


# In[80]:


similarity = cosine_similarity(vector)


# In[93]:


sorted(list(enumerate(similarity[3000])),reverse=True,key = lambda x: x[1])[1:6]


# In[105]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].id)
        


# In[106]:


recommend('Batman')


# In[99]:


import pickle


# In[100]:


pickle.dump(new,open('movie_list.pkl','wb'))


# In[101]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




