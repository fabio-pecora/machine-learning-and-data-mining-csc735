#!/usr/bin/env python
# coding: utf-8

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scatteer_matrix from pandas.plotting
import pandas as np
import pyplot from mayplotlib

# Load the Glass Identification Dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
glass_data = pd.read_csv(url, header=None)
glass_data.columns = ['Id', 'Refractive Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'Class']

# Drop the 'Id' column
glass_data = glass_data.drop('Id', axis=1)

#check the first few rows
print(glass_data.head())



# In[59]:


# part 3
# 6.1
# Pairplot scatter

sns.pairplot(glass_data, hue='Class')

plt.title('Pairplot of Glass Identification Attributes by Class')
plt.show()


# In[58]:


# 6.2 Count plot of the class variable
sns.countplot(x='Class', data=glass_data)
plt.title('Count of Glass Types in the Dataset')
plt.show()


# In[17]:


# Boxplot for each attribute
# visualize the spread and outliers 
plt.figure(figsize=(12, 6))
glass_data.drop('Class', axis=1).boxplot()
plt.title('Boxplot of Glass Identification Attributes')
plt.xticks(rotation=90)
plt.show()


# In[19]:


# Correlation matrix 
correlation_matrix = glass_data.corr()
# plot a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap of Glass Attributes')
plt.show()


# In[33]:


# Example of a relationship plot between Sodium and Refractive Index
sns.lmplot(x='Sodium', y='Refractive Index', hue='Class', data=glass_data)
plt.title('Relationship between Sodium and Refractive Index by Glass Class')
plt.show()


# In[35]:


# Relationship plot for Magnesium vs Aluminum
sns.lmplot(x='Magnesium', y='Aluminum', hue='Class', data=glass_data)
plt.title('Relationship between Magnesium and Aluminum by Glass Class')
plt.show()


# In[37]:


# Relationship plot for Calcium vs Barium
sns.lmplot(x='Calcium', y='Barium', hue='Class', data=glass_data)
plt.title('Relationship between Calcium and Barium by Glass Class')
plt.show()


# In[39]:


# Hexbin plot for Sodium vs Calcium
plt.figure(figsize=(10, 6))
plt.hexbin(glass_data['Sodium'], glass_data['Calcium'], gridsize=25, cmap='coolwarm')
plt.title('Hexbin Plot of Sodium vs Calcium')
plt.xlabel('Sodium')
plt.ylabel('Calcium')
plt.colorbar(label='Density')
plt.show()


# In[41]:


# Hexbin plot for Magnesium vs Aluminum
plt.figure(figsize=(10, 6))
plt.hexbin(glass_data['Magnesium'], glass_data['Aluminum'], gridsize=25, cmap='coolwarm')
plt.title('Hexbin Plot of Magnesium vs Aluminum')
plt.xlabel('Magnesium')
plt.ylabel('Aluminum')
plt.colorbar(label='Density')
plt.show()


# In[43]:


# Distribution plot for Refractive Index
sns.histplot(glass_data['Refractive Index'], kde=True)
plt.title('Distribution of Refractive Index in Glass Samples')
plt.xlabel('Refractive Index')
plt.ylabel('Frequency')
plt.show()



# In[45]:


# Distribution plot for Iron
sns.histplot(glass_data['Iron'], kde=True)
plt.title('Distribution of Iron in Glass Samples')
plt.xlabel('Iron')
plt.ylabel('Frequency')
plt.show()


# In[47]:


# Swarm plot for Calcium levels by glass class
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Class', y='Calcium', data=glass_data)
plt.title('Swarm Plot of Calcium Levels by Glass Class')
plt.xlabel('Glass Class')
plt.ylabel('Calcium')
plt.show()


# In[49]:


# Violin plot for Sodium levels by class
plt.figure(figsize=(10, 6))
sns.violinplot(x='Class', y='Sodium', data=glass_data)
plt.title('Violin Plot of Sodium Levels by Glass Class')
plt.xlabel('Glass Class')
plt.ylabel('Sodium')
plt.show()


# In[ ]:




