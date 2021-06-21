#!/usr/bin/env python
# coding: utf-8

# # About this dataframe
# ## Context
# Recorded crime for the Police Force Areas of England and Wales.
# The data are rolling 12-month totals, with points at the end of each financial year between year ending March 2003 to March 2007 and at the end of each quarter from June 2007.
# 
# ## Content
# The data are a single .csv file with comma-separated data.
# It has the following attributes:
# 
# 12 months ending: the end of the financial year.
# 
# PFA: the Police Force Area.
# 
# Region: the region that the criminal offence took place.
# 
# Offence: the name of the criminal offence.
# 
# Rolling year total number of offences: the number of occurrences of a given offence in the last year.
# 
# ## Source:
# https://www.kaggle.com/r3w0p4/recorded-crime-data-at-police-force-area-level

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# # Reading the dataset

# In[2]:


df = pd.read_csv("rec-crime-pfa.csv")
df.head()


# # Change the Column names for convinience

# In[3]:


mapper = {"Rolling year total number of offences":"num_offence",
         "Offence":"type_offence",
         "12 months ending":"date"}
df.rename(columns = mapper, inplace=True)


# # General Overview of Dataset

# In[4]:


print("The number of Features: {}".format(df.shape[1]))
print("The number of Observations: {}".format(df.shape[0]))


# In[5]:


df.info()


# In[6]:


df.describe()


# ## There is no Null values in this Dataset

# In[7]:


df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace = True)


# # Unique Values

# In[8]:


df.index.unique()


# In[9]:


df.Region.unique()


# In[10]:


df.PFA.unique()


# In[11]:


df.type_offence.unique()


# # We only consider the data after the year 2007

# In[12]:


df = df.loc[df.index>"01-01-2007"]


# In[13]:


df.loc[df.index.month == 3, "season"] = "Winter"
df.loc[df.index.month == 6, "season"] = "Summer"
df.loc[df.index.month == 9, "season"] = "Spring"
df.loc[df.index.month == 12, "season"] = "Fall"
df["year"] = df.index.year


# In[14]:


df.head()


# # Main Analysis

# ## Seasonal Crimes Analysis

# In[15]:


df_ys = pd.crosstab(index = df.year, columns = df.season,
                    values = df.num_offence, aggfunc = "mean",
                    margins = True, margins_name = "mean")
df_ys.head(15)


# ## As is shown, The mean amount of crime in different seasons are almost the same.

# In[16]:


plt.figure(figsize = [12,8])
plt.plot(df_ys.drop(index = 'mean').index,
         df_ys.drop(index = 'mean').loc[:,"mean"])


# In[17]:


df.groupby(["year","season"]).num_offence.mean().unstack().                                plot(kind = "bar", figsize = [12,8]);


# In[18]:


hue_order = ["Winter", "Spring", "Summer", "Fall"]
plt.figure(figsize = [12,8])
sns.barplot(data = df, x = 'year', y = "num_offence",
            hue = "season", dodge = True, hue_order = hue_order,);


# ## As is shown, the average number of crimes during the years 2007 and 2018 sunk from nearly 6000 to 4600 in 2013 and it increases up to more than 6000 again in 2018.

# # Regional Crime Analysis over the Years

# In[19]:


df_yr = pd.crosstab(index = df.year, columns = df.Region,
                    values = df.num_offence, aggfunc = "mean",
                    margins = True, margins_name = "mean")
df_yr.head(15)


# ## We should Ommit Three columns namely, Fraud: Action Fraud, Fraud: CIFAS,	Fraud: UK Finance, Because these parts are related to the type of offence, not the region of the crime!!

# In[20]:


df_yr.drop(columns = ['Fraud: Action Fraud','Fraud: CIFAS', 'Fraud: UK Finance'],
          index = 'mean').plot(figsize = [12,8])


# ## Among all regions Action Fraud and CIFAS have been significantly rose during the years 2011 and 2018. The rest of regions saw a slight decrease and then increase.
# ## Another fact about this plot is that the most criminal region in the whole analyzed area is London. The amount of crime in the other regions of UK is pretty much the same.

# # Type of Offence Analysis Over the Years

# In[21]:


df_yt = pd.crosstab(index = df.year, columns = df.type_offence,
                    values = df.num_offence, aggfunc = "mean",
                    margins = True, margins_name = "mean")
df_yt.head(15)


# ## Since there are some Null Values in each column, we consider backward fill value method here.
# ## Moreover, Three columns of Action Fraud, CIFAS and UK Finance are also seperated, since the amount of these crimes are much more than the rest and their analysis cannot be shown in the plot vividly.

# In[22]:


blocked_list = ["Action Fraud","CIFAS","UK Finance"]
df_yt.fillna(method = "bfill").drop(columns = blocked_list, index = 'mean').                                plot(figsize = [12,80], subplots = True,grid = True);


# ## Most highlights during the years 2007 to 2018:
# 
# 1) Bicycle Theft: has decreased from 2500 in 2007 to nearly 2000 in 2016. afterwards however, it increased up to 2200 in 2018
# 
# 2) Criminal Damage and Ason decreased from 25000 to 12500. This is nearly a 50% decrease.
# 
# 3) Death or serious injury caused by illegal driving rose from 10 to 16. This amount is too little to be analyzed.
# 
# 4) Domestic Burglury has decreased slightly from 6000 in 2007 to 4000 in 2016. This amount has fallen significantly to zero in 2018. This is strange, there must be missing values for this crime in our data set in the year 2018.
# 
# 5) Drug offence rose to 5500 and decreased to lower than 3500 in 2018
# 
# 6) Homicide decreased from 17 in 2007 to 12 in 2014. then it increased to 16 in 2018.
# 
# 7) Miscellaneous Crime against society decreased slightly from 1400 in 2007 to 1000 in 2013 and increased to 2200 in 2018.
# 
# 8) Non domestic Burglury decreased steadily from 7000 in 2007 to nearly 4500 in 2016. Afterwards it decreased significantly to zero in 2018.
# 
# 9) Between the years 2016 and 2018 however, the non residential burglury increased remarkably to nearly 3000.
# 
# 10) Possession of weapons offences was nearly 900 in 2007. it saw a steady decrease and reached its minimum of below 500 in 2013 and it rose again up to more than 900 in 2018.
# 
# 11)Public oreder offences showed a slight decrease from 5500 in 2007 to 3500 in 2013 and it increased to more than 9000 in 2018.
# 
# 12)Robbery decreased from 2200 to 1200 in 2015 and it increased to 1800 in 2018.
# 
# 13)Unfortunately, The sexual offences increased steadily during this period of time from lower than 1500 in 2007 to more than 3500 in 2018. Looking at this statistics from another perspective, during the past years, with the remarkable growth of social media, people are not affraid or ashamed of disclosing sexual abuses. Maybe this statistics can be reffered to the number of disclosures of this crime to the police, rather than the number of sexual herasments which actually were occured. Hope So!
# 
# 14) The same thing could be true for Stalking and Herassments, which showed a great increase from lower than 2000 in 2007 to 8000 in 2018.
# 
# 15)The theft from people fluctuated between 1800 and 2200
# 
# 16) Vehicle offences decreased from 16000 in 2007 to 8000 in 2015. Subsequently, it increased to 10000 in 2018.
# 
# 17)Violence with injuries decreased from 11000 in 2007 to 7000 in 2013 and it increased to more than 12000 in 2018.
# 
# 18) Violence without injuries however, increased significantly from lower than 6000 to more than 14000.
# 
# # The Average incidence of crimes in all of the aforementioned areas decreased from 2007 to 2013. The years 2013 to 2015 showed a plateau of the minimum incidences and afterwards the average occurace of crimes again increased significantly.
# 

# In[23]:


selected_list = ["Action Fraud","CIFAS","UK Finance"]
df_yt.fillna(method = "bfill").loc[:,selected_list].drop(index = 'mean').                                plot(figsize = [12,10], subplots = True,grid = True);


# ## Most highlights during the years 2007 to 2018:
# 
# 1) Action Fraud increased from lower than 50,000 in 2011 to nearly 300,000 in 2018.
# 
# 2) CIFAS showed a similar behavior and reached its maximum of 300,000 in years 2015 and 2016 and it fell slightly to 280,000 in 2018.
# 
# 3) UK Finance however decreased from 120,000 in 2012 to 80,000 in 2018. This crime is an exceptional case and as opposed to all other crimes it fell during this period of time.

# # Regional Analysis of Crimes based on the Type of Crime

# In[24]:


blocked_index = ["Fraud: Action Fraud","Fraud: CIFAS","Fraud: UK Finance"]
blocked_columns = ["Action Fraud","CIFAS","UK Finance","mean"]
df_rt = pd.crosstab(index = df.Region, columns = df.type_offence,
                    values = df.num_offence, aggfunc = "mean",
                    margins = True, margins_name = "mean").\
                    drop(index = blocked_index, columns = blocked_columns)
df_rt.head(15)


# In[25]:


df_rt.T.plot(kind = 'barh', figsize = [12,150], subplots = True,
             grid = True, fontsize = 15, sharex = False, stacked = True);


# ## Highlights:
# 
# 1) The most common crimes in all regions are "Criminal Damage and Arson", "Violence with and without Injury", "Vehicle Offences" and "Shoplifting".
# Almost all of the regions are showing these crimes to be the most common.
# 2) The least common crimes are "Illegal Driving", "Possession of Weapon" and "Homocide".
