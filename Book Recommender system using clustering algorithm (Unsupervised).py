#!/usr/bin/env python
# coding: utf-8

# # Recommender system types:-
# 
# # -> Content Based filtering - Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.
# 
# # -> Collaboartive filtering -  collaborative filtering uses similarities between users and items simultaneously to provide recommendations. This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user A based on the interests of a similar user B. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features.
# 
# # Youtube uses collaborative filtering
# 
# # -> Hybrid filtering - Mix of content based and collaborative filtering
# # e.g. Facebook

# In[1]:


import pandas as pd
import pickle 
import numpy as np

# Load data from the pickle file
#with open(r"C:\Users\rimjh\Downloads\pt.pkl", 'rb') as f:
   # data = pickle.load(f)

# Convert the data into a DataFrame
#df = pd.DataFrame(data)

# Write DataFrame to CSV file
#df.to_csv('data.csv', index=False)


# In[2]:


#with open(r"C:\Users\rimjh\Downloads\books.pkl", 'rb') as f:
 #   data1 = pickle.load(f)

# Convert the data into a DataFrame
#df1 = pd.DataFrame(data1)

# Write DataFrame to CSV file
#df1.to_csv('data1.csv', index=False)


# In[3]:


#with open(r"C:\Users\rimjh\Downloads\popular.pkl", 'rb') as f:
 #   data2 = pickle.load(f)

# Convert the data into a DataFrame
#df2 = pd.DataFrame(data)

# Write DataFrame to CSV file
#df2.to_csv('data2.csv', index=False)


# In[4]:


import pandas as pd


# In[5]:


books = pd.read_csv("D:\downloads\Books.csv\Books.csv" , error_bad_lines = False  ) #error_bad_lines = False skips lines which can cause error


# In[6]:


books.head()


# In[7]:


books.columns


# In[8]:


books=books[["ISBN","Book-Title","Book-Author","Year-Of-Publication","Publisher"]]


# In[9]:


books.head()


# In[10]:


books.isnull().sum()


# In[11]:


books.duplicated().sum()


# In[12]:


books.rename(columns={"Book-Title":"Title","Book-Author":"Author","Year-Of-Publication":"Year"},inplace=True)


# In[13]:


books.head()


# In[14]:


users=pd.read_csv(r"D:\downloads\archive\Users.csv" , error_bad_lines = False  )


# In[15]:


users.head()


# In[16]:


users.rename(columns={"User-ID":"ID"},inplace=True)


# In[17]:


users.isnull().sum()


# In[18]:


users.duplicated().sum()


# In[19]:


ratings = pd.read_csv(r"D:\downloads\archive\Ratings.csv", error_bad_lines = False )


# In[20]:


ratings.head()


# In[21]:


ratings.rename(columns={"User-ID":"ID","Book-rating":"Rating"})


# In[22]:


ratings.head()


# In[23]:


#ratings.iloc[:,0:2].head()


# In[24]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[25]:


ratings['User-ID'].value_counts()


# In[26]:


ratings['User-ID'].value_counts().shape  # Those who rated 


# In[27]:


o=ratings['User-ID'].value_counts()>250
o


# In[28]:


#ratings[o]


# In[29]:


o[o] # Gives the value which have true values


# # Popularity Based Recommender System

# In[30]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[31]:


ratings_with_name 


# In[32]:


ratings_with_name.shape


# In[33]:


num_rating = ratings_with_name.groupby('Title')


# In[34]:


num_rating


# In[35]:


#num_rating = ratings_with_name.groupby('Title').counts()['Book-Rating'].reset_index()
num_rating = ratings_with_name.groupby('Title')['Book-Rating'].count().reset_index()


# In[36]:


num_rating


# In[37]:


num_rating.rename(columns={'Book-Rating':'Count_ratings'},inplace=True)


# In[38]:


num_rating


# In[39]:


num_rating.shape


# In[40]:


avg_rating = ratings_with_name.groupby('Title')['Book-Rating'].mean().reset_index()


# In[41]:


avg_rating.rename(columns={'Book-Rating':'Avg_ratings'},inplace=True)


# In[42]:


avg_rating.shape


# In[43]:


avg_rating


# In[44]:


popular_df = num_rating.merge(avg_rating,on="Title")


# In[45]:


popular_df


# In[46]:


popular_df = popular_df[popular_df['Count_ratings']>=200].sort_values('Avg_ratings',ascending=False)


# In[47]:


popular_df.head(10)


# In[48]:


popular_df = popular_df.merge(books,on='Title').drop_duplicates('Title')[['Title',"Author",'Count_ratings','Avg_ratings']] 


# In[49]:


popular_df.head()


# In[50]:


#popular_df['Image-URL-M'][0]


# # Collaborative Filtering Based Recommender System

# In[51]:


y = ratings_with_name.groupby('User-ID')


# In[52]:


y.head()


# In[53]:


y = ratings_with_name.groupby('User-ID').count()


# In[54]:


y.head()


# In[55]:


y = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200


# In[56]:


y.head()


# In[57]:


users_with_who_rated_more = y[y].index


# In[58]:


users_with_who_rated_more  # These are the indices of users who rated more than 200 that means they have value true 


# In[59]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(users_with_who_rated_more)]


# In[60]:


filtered_rating


# In[61]:


y = filtered_rating.groupby('Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[62]:


final_ratings = filtered_rating[filtered_rating['Title'].isin(famous_books)]


# In[63]:


final_ratings


# In[64]:


conc = final_ratings.pivot_table(index='Title',columns='User-ID',values='Book-Rating')


# In[65]:


conc


# In[66]:


conc.fillna(0,inplace=True)


# In[67]:


conc


# In[68]:


from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(conc)
similarity_scores.shape


# In[69]:


def recommend(book_name):
    # index fetch
    index = np.where(conc.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Title'] == conc.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Title')['Title'].values))
        item.extend(list(temp_df.drop_duplicates('Title')['Author'].values))
       #item.extend(list(temp_df.drop_duplicates('Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data


# In[73]:


recommend('1984')


# In[71]:


conc.index[545]


# In[74]:


recommend("2nd Chance")


# In[82]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[ ]:





# In[83]:


books.drop_duplicates('Title')


# In[85]:


pickle.dump(conc,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))
 


# In[ ]:




