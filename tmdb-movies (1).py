#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series,DataFrame
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movies_dframe = pd.read_csv('https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd1c4c_tmdb-movies/tmdb-movies.csv')


# ### Data Wrangling

# ##### Get a quick overview of the dataset

# In[3]:


movies_dframe.info()


# In[4]:


movies_dframe.shape


# ### Cleaning Data

# ### Deleting the unused columns on our project

# In[5]:


#creating a list of the required deleted columns
del_col = [ 'id', 'imdb_id', 'popularity', 'budget_adj', 'revenue_adj', 'homepage', 'keywords', 'overview', 'production_companies', 'vote_count', 'vote_average']


# In[6]:


#removing the list we have just created with .drop()
movies_dframe = movies_dframe.drop(del_col, 1)


# In[7]:


movies_dframe


# In[8]:


#keep the first one and delete duplicate rows
movies_dframe.drop_duplicates(keep = 'first', inplace = True)


# In[9]:


movies_dframe.shape


#  replace the value of '0' to NaN

# In[10]:


#creating a list with the required columns need to be checked.
check_row = ['budget', 'revenue']
#replace the value of '0' to NaN for the list we have just created.
movies_dframe[check_row] = movies_dframe[check_row].replace(0, np.NaN)
#delete or drop any nan values on the selected columns list
movies_dframe.dropna(subset = check_row, inplace = True)
movies_dframe.shape


# replacing 0 with NaN of runtime column

# In[11]:


#replacing 0 with NaN of runtime column
movies_dframe['runtime'] = movies_dframe['runtime'].replace(0, np.NaN)


# convert the 'release_date' column to date format

# In[12]:


#convert the 'release_date' column to date format
movies_dframe.release_date = pd.to_datetime(movies_dframe['release_date'])
movies_dframe.head(2)


# convert the 'budget'& 'revenue'column to int64

# In[13]:


#creating a list with the required columns need to be datatype changed.
change_coltype = ['budget', 'revenue']
movies_dframe[change_coltype] = movies_dframe[change_coltype].applymap(np.int64)
movies_dframe.dtypes


# In[14]:


#renames the columns with .rename function by dealing with the column name a dictionary key
movies_dframe.rename(columns = {'budget' : 'budget_(US-$)', 'revenue' : 'revenue_(US-$)'}, inplace = True)


# In[15]:


movies_dframe.head(1)


# the profits of each movie

# In[28]:


movies_dframe.insert(2, 'profit_(in_US_Dollars)', movies_dframe['revenue_(US-$)'] - movies_dframe['budget_(US-$)'])
movies_dframe['profit_(in_US_Dollars)'] = movies_dframe['profit_(in_US_Dollars)'].apply(np.int64)
movies_dframe.head(1)


# ### Q1 Which movie earns the most and least profit?

# In[29]:


#create a function shows thelowest and highest value of columns with the column name as an argument
def highest_lowest(column_name):
    highest_id = movies_dframe[column_name].idxmax()
    highest_details = pd.DataFrame(movies_dframe.loc[highest_id])
    lowest_id = movies_dframe[column_name].idxmin()
    lowest_details = pd.DataFrame(movies_dframe.loc[lowest_id])
    two_in_one_data = pd.concat([highest_details, lowest_details], axis = 1)
    
    return two_in_one_data

highest_lowest('profit_(in_US_Dollars)')


# ### Q2 Which movie had the greatest and least runtime?

# In[20]:


highest_lowest('runtime')


# ### Q3 Which movie had the greatest and least budget?

# In[21]:


highest_lowest('budget_(in_US-Dollars)')


# ### Q4 Which movie had the greatest and least budget?

# In[22]:


highest_lowest('revenue_(in_US-Dollars)')


# ### Q5 What is the average runtime of all movies?

# In[23]:


def average_func(column_name):
    
    return movies_dframe[column_name].mean()


# In[24]:


average_func('runtime')


# ##### The average runtime of all movies in this dataset is 109 mins approx. We want to get a deeper look and understanding of runtime of all movies so Let's plot it.

# In[25]:


sns.set_style('darkgrid')
plt.rc('xtick', labelsize = 5)
plt.rc('ytick', labelsize = 5)
plt.figure(figsize=(9,6), dpi = 100)
plt.xlabel('Runtime of Movies', fontsize = 20)
plt.ylabel('Number of Movies', fontsize=20)
plt.title('Runtime distribution of all the movies', fontsize=14)
plt.hist(movies_dframe['runtime'], rwidth = 0.9, bins =31)
plt.show()


# In[26]:


movies_dframe['runtime'].describe()


# ### Q6 in which year we had the most movies making profits?

# In[27]:


profits = movies_dframe.groupby('release_year')['profit_(in_US_Dollars)'].sum()
plt.figure(figsize=(18,9), dpi = 130)
plt.xlabel('Release Year of Movies', fontsize = 20)
plt.ylabel('Total Profits made by Movies', fontsize = 20)
plt.title('Calculating Total Profits made by all movies in year which it released.')
plt.plot(profits)
plt.show()


# In[ ]:


profits.idxmax()


# In[ ]:


profits = pd.DataFrame(profits)


# In[ ]:


profits.tail()


# ### Q7 Average runtime of movies

# In[ ]:


profit_movies_dframe = movies_dframe[movies_dframe['profit_(in_US_Dollars)'] >= 50000000]
profit_movies_dframe.index = range(len(profit_movies_dframe))
profit_movies_dframe.index = profit_movies_dframe.index + 1
profit_movies_dframe.head(2)


# In[ ]:


len(profit_movies_dframe)


# In[ ]:


def prof_avg_fuc(column_name):
    return profit_movies_dframe[column_name].mean()


# In[ ]:


prof_avg_fuc('runtime')


# ### Average Budget

# In[ ]:


prof_avg_fuc('budget_(in_US-Dollars)')


# ### Average Revenue of Movies

# In[ ]:


prof_avg_fuc('revenue_(in_US-Dollars)')


# ### Average Profit of Movies

# In[ ]:


prof_avg_fuc('profit_(in_US_Dollars)')


# ### Which directer directed most films?

# In[ ]:


def extract_data(column_name):
    all_data = profit_movies_dframe[column_name].str.cat(sep = '|')
    all_data = pd.Series(all_data.split('|'))
    count = all_data.value_counts(ascending = False)
    
    return count


# In[ ]:


director_count = extract_data('director')
director_count.head()


# ### most cast appeared

# In[ ]:


cast_count = extract_data('cast')
cast_count.head()


# ### Most genre produced

# In[ ]:


genre_count = extract_data('genres')
genre_count.head()


# In[ ]:


genre_count.sort_values(ascending = True, inplace = True)
ax = genre_count.plot.barh(color = '#007482', fontsize = 15)
ax.set(title = 'The Most filmed genres')
ax.set_xlabel('Number of Movies', color = 'g', fontsize = '18')
ax.figure.set_size_inches(12, 10)
plt.show()


# In[ ]:




