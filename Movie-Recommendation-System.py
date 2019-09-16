#!/usr/bin/env python
# coding: utf-8

# - This notebook implements a movie recommender system. 
# - Recommender systems are used to suggest movies or songs to users based on their interest or usage history. 
# - For example, Netflix recommends movies to watch based on the previous movies you've watched.  
# - In this example, we will use Item-based Collaborative Filter 
# 
# 
# 
# - Dataset MovieLens: https://grouplens.org/datasets/movielens/100k/ 
# 

# # STEP #0: LIBRARIES IMPORT
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## **The below Code Snippet is used to set connection between Google drive and Google Colab to load multiple files (Optional)**

# In[3]:


# Install a Drive FUSE wrapper.
# https://github.com/astrada/google-drive-ocamlfuse
get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')



# Generate auth tokens for Colab
from google.colab import auth
auth.authenticate_user()


# Generate creds for the Drive FUSE library.
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
vcode = getpass.getpass()
get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')


# Create a directory and mount Google Drive using that directory.
get_ipython().system('mkdir -p My Drive')
get_ipython().system('google-drive-ocamlfuse My Drive')


get_ipython().system('ls My Drive/')

# Create a file in Drive.
get_ipython().system('echo "This newly created file will appear in your Drive file list." > My Drive/created.txt')


# # STEP #1: IMPORT DATASET

# In[4]:


# Two datasets are available, let's load the first one:
movie_titles_df = pd.read_csv("./dataset/Movie_Id_Titles")
movie_titles_df.head(20)


# In[ ]:


# Let's load the second one!
movies_rating_df = pd.read_csv('./dataset/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])


# In[7]:


movies_rating_df.head(10)


# In[8]:


movies_rating_df.tail()


# In[ ]:


# Let's drop the timestamp 
movies_rating_df.drop(['timestamp'], axis = 1, inplace = True)


# In[10]:


movies_rating_df


# In[11]:


movies_rating_df.describe()


# In[12]:


movies_rating_df.info()


# In[ ]:


# Let's merge both dataframes together so we can have ID with the movie name
movies_rating_df = pd.merge(movies_rating_df, movie_titles_df, on = 'item_id') 


# In[14]:


movies_rating_df


# In[15]:


movies_rating_df.shape


# # STEP #2: VISUALIZE DATASET

# In[16]:


movies_rating_df.groupby('title')['rating'].describe()


# In[ ]:


ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']


# In[ ]:


ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']


# In[19]:


ratings_df_count


# In[ ]:


ratings_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)


# In[21]:


ratings_mean_count_df.reset_index()


# In[22]:


ratings_mean_count_df['mean'].plot(bins=100, kind='hist', color = 'r') 


# In[23]:


ratings_mean_count_df['count'].plot(bins=100, kind='hist', color = 'r') 


# In[24]:


# Let's see the highest rated movies!
# Apparently these movies does not have many reviews (i.e.: small number of ratings)
ratings_mean_count_df[ratings_mean_count_df['mean'] == 5]


# In[25]:


# List all the movies that are most rated
# Please note that they are not necessarily have the highest rating (mean)
ratings_mean_count_df.sort_values('count', ascending = False).head(100)


# # STEP #3: PERFORM ITEM-BASED COLLABORATIVE FILTERING ON ONE MOVIE SAMPLE

# In[ ]:


userid_movietitle_matrix = movies_rating_df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')


# In[27]:


userid_movietitle_matrix


# In[ ]:


titanic = userid_movietitle_matrix['Titanic (1997)']


# In[29]:


titanic


# In[30]:


# Let's calculate the correlations
titanic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns=['Correlation'])
titanic_correlations = titanic_correlations.join(ratings_mean_count_df['count'])


# In[31]:


titanic_correlations


# In[32]:


titanic_correlations.dropna(inplace=True)
titanic_correlations


# In[33]:


# Let's sort the correlations vector
titanic_correlations.sort_values('Correlation', ascending=False)


# In[34]:


titanic_correlations[titanic_correlations['count']>80].sort_values('Correlation',ascending=False).head()


# # STEP#4: CREATE AN ITEM-BASED COLLABORATIVE FILTER ON THE ENTIRE DATASET 

# In[35]:


# Recall this matrix that we created earlier of all movies and their user ID/ratings
userid_movietitle_matrix


# In[ ]:


movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)
# pearson : standard correlation coefficient
# Obtain the correlations between all movies in the dataframe


# In[37]:


movie_correlations


# In[ ]:


# Let's create our own dataframe with our own ratings!
myRatings = pd.read_csv("Drive/Project 8/My_Ratings.csv")
#myRatings.reset_index


# In[40]:


myRatings


# In[41]:


len(myRatings.index)


# In[42]:


myRatings['Movie Name'][0]


# In[ ]:


similar_movies_list = pd.Series()
for i in range(0, 2):
    similar_movie = movie_correlations[myRatings['Movie Name'][i]].dropna() # Get same movies with same ratings
    similar_movie = similar_movie.map(lambda x: x * myRatings['Ratings'][i]) # Scale the similarity by your given ratings
    similar_movies_list = similar_movies_list.append(similar_movie)


# In[44]:


similar_movies_list.sort_values(inplace = True, ascending = False)
print (similar_movies_list.head(10))


# In[ ]:




