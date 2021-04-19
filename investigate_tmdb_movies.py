#!/usr/bin/env python
# coding: utf-8

# ## Questions to be Answered

# ####  Q1 Which movie earns the most and least profit?
# ####  Q2 Which movie had the greatest and least runtime?
# ####  Q3 Which movie had the greatest and least budget?
# ####  Q4 Which movie had the greatest and least REVENUE?
# ####  Q5 What is the average runtime of all movies?
# ####  Q6 in which year we had the most movies making profits?
# ####  Q7 Average runtime of movies
# ####  Q8 Average Budget
# ####  Q9 Average Revenue of Movies
# ####  Q10 Average Profit of Movies
# ####  Q11 Which directer directed most films?
# ####  Q12 most cast appeared
# ####  Q13 Most genre produced

# In[54]:


import pandas as pd
from pandas import Series,DataFrame
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


movies_dframe = pd.read_csv('https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd1c4c_tmdb-movies/tmdb-movies.csv')


# ### Data Wrangling

# ##### Get a quick overview of the dataset

# In[56]:


movies_dframe.info()


# In[57]:


movies_dframe.shape


# ### Cleaning Data

# ### Deleting the unused columns on our project

# In[58]:


#creating a list of the required deleted columns
del_col = [ 'id', 'imdb_id', 'popularity', 'budget_adj', 'revenue_adj', 'homepage', 'keywords', 'overview', 'production_companies', 'vote_count', 'vote_average']


# In[59]:


#removing the list we have just created with .drop()
movies_dframe = movies_dframe.drop(del_col, 1)


# In[60]:


movies_dframe


# In[61]:


#keep the first one and delete duplicate rows
movies_dframe.drop_duplicates(keep = 'first', inplace = True)


# In[62]:


movies_dframe.shape


#  replace the value of '0' to NaN

# In[63]:


#creating a list with the required columns need to be checked.
check_row = ['budget', 'revenue']
#replace the value of '0' to NaN for the list we have just created.
movies_dframe[check_row] = movies_dframe[check_row].replace(0, np.NaN)
#delete or drop any nan values on the selected columns list
movies_dframe.dropna(subset = check_row, inplace = True)
movies_dframe.shape


# replacing 0 with NaN of runtime column

# In[64]:


#replacing 0 with NaN of runtime column
movies_dframe['runtime'] = movies_dframe['runtime'].replace(0, np.NaN)


# convert the 'release_date' column to date format

# In[65]:


#convert the 'release_date' column to date format
movies_dframe.release_date = pd.to_datetime(movies_dframe['release_date'])
movies_dframe.head(2)


# convert the 'budget'& 'revenue'column to int64

# In[66]:


#creating a list with the required columns need to be datatype changed.
change_coltype = ['budget', 'revenue']
movies_dframe[change_coltype] = movies_dframe[change_coltype].applymap(np.int64)
movies_dframe.dtypes


# In[67]:


#renames the columns with .rename function by dealing with the column name a dictionary key
movies_dframe.rename(columns = {'budget' : 'budget_(in_US-Dollars)', 'revenue' : 'revenue_(in_US-Dollars)'}, inplace = True)


# In[68]:


movies_dframe.head(1)


# the profits of each movie

# In[69]:


#create a new column with the profit of each movie
movies_dframe.insert(2, 'profit_(in_US_Dollars)', movies_dframe['revenue_(in_US-Dollars)'] - movies_dframe['budget_(in_US-Dollars)'])
#change the data type to int of the column "profit_(in_US_Dollars"
movies_dframe['profit_(in_US_Dollars)'] = movies_dframe['profit_(in_US_Dollars)'].apply(np.int64)
movies_dframe.head(1)


# ### Q1 Which movie earns the most and least profit?

# In[70]:


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

# In[71]:


highest_lowest('runtime')


# ### Q3 Which movie had the greatest and least budget?

# In[72]:


highest_lowest('budget_(in_US-Dollars)')


# ### Q4 Which movie had the greatest and least REVENUE?

# In[73]:


highest_lowest('revenue_(in_US-Dollars)')


# ### Q5 What is the average runtime of all movies?

# In[74]:


#calculates average of a column with the column name as an argument
def average_func(column_name):
    
    return movies_dframe[column_name].mean()


# In[75]:


average_func('runtime')


# ##### The average runtime of all movies in this dataset is 109 mins approx. We want to get a deeper look and understanding of runtime of all movies so Let's plot it.

# In[76]:


#a histogram of runtime of movies
sns.set_style('darkgrid')
plt.rc('xtick', labelsize = 5)
plt.rc('ytick', labelsize = 5)
plt.figure(figsize=(9,6), dpi = 100)
plt.xlabel('Runtime of Movies', fontsize = 20)
plt.ylabel('Number of Movies', fontsize=20)
plt.title('Runtime distribution of all the movies', fontsize=14)
plt.hist(movies_dframe['runtime'], rwidth = 0.9, bins =31)
plt.show()


# ##### as you can see the tallest bar here is time interval between 85-100 . and around 1000 movies out of 3855 movies have the runtime between these time intervals. So we can also say from this graph that mode time of movies is around 85-110 min, has the highest concentration of data points around this time interval.

# In[77]:


#getting specific runtime points at x positions
movies_dframe['runtime'].describe()


# ### Q6 in which year we had the most movies making profits?

# In[78]:


#plot used to know the profits of movies for every year
#the groupby. function to calculate and show all the movies for that year and also the profits for that years
profits = movies_dframe.groupby('release_year')['profit_(in_US_Dollars)'].sum()
plt.figure(figsize=(18,9), dpi = 130)
plt.xlabel('Release Year of Movies', fontsize = 20)
plt.ylabel('Total Profits made by Movies', fontsize = 20)
plt.title('Calculating Total Profits made by all movies in year which it released.')
plt.plot(profits)
plt.show()


# ##### The year 2015, shows us the highest peak, having the highest profit than in any year, of more than 18 billion dollars. This graph doesn't exactly prove us that every year pass by, the profits of movies will increase but when we see in terms of decades it does show significant uprise in profits. At the year 2000, profits were around 8 biilion dollars, but in just 15 years it increased by 10+ biilion dollars. Last 15 years had a significant rise in profits compared to any other decades as we can see in the graph.

# In[79]:


#the year with the highest profit
profits.idxmax()


# In[80]:


#storing the value
profits = pd.DataFrame(profits)


# In[81]:


profits.tail()


# ### Q7 Average runtime of movies

# In[82]:


#creating new dataframe with movies having profit $50M or more
profit_movies_dframe = movies_dframe[movies_dframe['profit_(in_US_Dollars)'] >= 50000000]
profit_movies_dframe.index = range(len(profit_movies_dframe))
profit_movies_dframe.index = profit_movies_dframe.index + 1
profit_movies_dframe.head(2)


# In[83]:


#number of rows of a the new dataframe
len(profit_movies_dframe)


# In[84]:


#a new average function for the new dataset
def prof_avg_fuc(column_name):
    return profit_movies_dframe[column_name].mean()


# In[85]:


# calling the new average function
prof_avg_fuc('runtime')


# ### Average Budget

# In[86]:


# calling the new average function
prof_avg_fuc('budget_(in_US-Dollars)')


# ### Average Revenue of Movies

# In[87]:


# calling the new average function
prof_avg_fuc('revenue_(in_US-Dollars)')


# ### Average Profit of Movies

# In[88]:


# calling the new average function
prof_avg_fuc('profit_(in_US_Dollars)')


# ### Which directer directed most films?

# In[89]:


#function which will take any column as argument from which data is need to be extracted
def extract_data(column_name):
    all_data = profit_movies_dframe[column_name].str.cat(sep = '|')
    all_data = pd.Series(all_data.split('|'))
    count = all_data.value_counts(ascending = False)
    
    return count


# In[90]:


#calling the new function to show  which directer directed most films
director_count = extract_data('director')
director_count.head()


# ### most cast appeared

# In[91]:


#calling the new function to show which most cast appeared
cast_count = extract_data('cast')
cast_count.head()


# ### Most genre produced

# In[92]:


#calling the new function to show which Most genre produced
genre_count = extract_data('genres')
genre_count.head()


# In[93]:


#plot used to know Which genre were more successful and give a profit more than 50 million profit
genre_count.sort_values(ascending = True, inplace = True)
ax = genre_count.plot.barh(color = '#007482', fontsize = 15)
ax.set(title = 'The Most filmed genres')
ax.set_ylabel('Genre of Movies', color = 'g', fontsize = '18')
ax.set_xlabel('Number of Movies', color = 'g', fontsize = '18')
ax.figure.set_size_inches(12, 10)
plt.show()


# #### here is a plot with the movies that achieved more than 50M profit .comedy came on the first place  with more than 490 movies achieved this creiteria,the drama came second and the foreign on the last place with just one movie

# ### Conclusion
# As i have answered the questions that i thought would be interesting, i want to wrap up all my findings in this way
# 
# Choose any director from this - Steven Spielberg, Robert Zemeckis, Ron Howard, Tony Scott, Ridley Scott.
# 
# Choose any cast from this - Actors - Tom Cruise, Brad Pitt, Tom Hanks, Sylvester Stallone, Denzel Washington.
# 
# Choose these genre - Action, Adventure, Thriller, Comedy, Drama.
# 

# #### Limitations
# It's not absolutly 100 % guaranteed analysis ,But it shows us that we have a very high probability of making high REVENUE . All these directors, actors, genres and released dates have a common trend of attraction. If we release a movie with these factors, the result could be a high expectations from this movie with audiance.Even if the movie was worth, people's high expectations would lead in results ultimately effecting the profits.This was just one example of an influantial factor that would lead to different results, there are many that have to be taken care of. Some limitations the dataset contains are null and zero values in some features . For example null values is an obstacle which stopped me when I was analyzing the top casted actors.
# 

# In[ ]:




