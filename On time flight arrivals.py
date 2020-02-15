#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Building machine learning to predict whether a flight would be delayed or not


# In[ ]:





# In[1]:


get_ipython().system(u'curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv')


# In[2]:


import pandas as pd
df = pd.read_csv("flightdata.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().values.any()


# In[5]:


df.isnull().sum()


# In[6]:


df = df.drop('Unnamed: 25', axis=1)
df.isnull().sum()


# In[7]:


df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
df.isnull().sum()


# In[8]:


df[df.isnull().values.any(axis=1)].head()


# In[9]:


df = df.fillna({'ARR_DEL15': 1})


# In[10]:


df.iloc[177:190]


# In[11]:


#perfomming binning to reduce the value of the CRS_DEP_TIME by 100
#Its a form of quantization and rounds off the number to the nearest integer
import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME']/100)
df.head()


# In[12]:


df=pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()


# In[13]:


#Splitting the data using sklearn
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)


# In[14]:


train_x.shape


# In[15]:


test_x.shape


# In[16]:


train_y.shape


# In[17]:


test_y.shape


# In[18]:


#Preventing overfitting using random forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)


# In[19]:


#making the predictions and getting the score
predicted = model.predict(test_x)
model.score(test_x, test_y)


# In[20]:


#generating the prediction probabilities from the test data
from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test_x)


# In[21]:


#generate an ROC AUC score from the probabilities using scikit-learn's roc_auc_score method
roc_auc_score(test_y, probabilities[:, 1])


# In[22]:


#Confusion matrix to get false positives, false negatives, true positives and true negatives
#The first row in the output represents flights that were on time. 
#The first column in that row shows how many flights were correctly predicted to be on time, 
#while the second column reveals how many flights were predicted as delayed but weren't. 
#From this, the model appears to be adept at predicting that a flight will be on time.
# second row, which represents flights that were delayed. 
#The first column shows how many delayed flights were incorrectly predicted to be on time. 
#The second column shows how many flights were correctly predicted to be delayed. Clearly, 
#the model isn't nearly as adept at predicting that a flight will be delayed as it is at predicting that a flight will arrive on time.
#What you want in a confusion matrix is large numbers in the upper-left and lower-right corners, 
#and small numbers (preferably zeros) in the upper-right and lower-left corners.

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predicted)


# In[24]:


#other measures of accuracy for a classification model include precision and recall. 
#Using precision to predict precision of the model
from sklearn.metrics import precision_score

train_predictions = model.predict(train_x)
precision_score(train_y, train_predictions)


# In[25]:


#Measuring model recall using skelean's recall_score
from sklearn.metrics import recall_score

recall_score(train_y, train_predictions)


# In[ ]:





# In[26]:


#Visualizing modl output

#Using matplotlib


# In[29]:


get_ipython().magic(u'matplotlib inline')
#It is a magic function that renders the figure in a notebook (instead of displaying a dump of the figure object)
#Graph is saved in the notebook
#It enables Jupyter to render Matplotlib output in a notebook without making repeated calls to show the graphs
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[30]:


#To see Matplotlib at work, execute the following code in a new cell to plot the ROC curve for the machine-learning model you built


# In[31]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# In[33]:


#Function to call the ml model earlier built


# In[34]:


def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]


# In[35]:


#Date input to the predict_delay function use the international date format dd/mm/year.
predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')


# In[36]:


#Modify the code to compute the probability that the same flight a day later will arrive on time:
#Show model recall
predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL')


# In[37]:


#compute the probability that a morning flight the same day from Atlanta to Seattle will arrive on time:
predict_delay('2/10/2018 10:00:00', 'ATL', 'SEA')


# In[38]:


#Predicting a number of days
import numpy as np

labels = ('Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7')
values = (predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('3/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('4/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('5/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('6/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('7/10/2018 21:45:00', 'JFK', 'ATL'))
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))


# In[42]:


#probability that flights leaving SEA for ATL at 9:00 a.m., noon, 3:00 p.m., 6:00 p.m., and 9:00 p.m. on January 30 will arrive on time


# In[43]:


import numpy as np

labels = ('9:00 am', 'noon', '3:00 pm', '6:00 pm', '9:00 pm')
values = (predict_delay('30/1/2018 09:00:00', 'SE', 'ATL'),
          predict_delay('30/1/2018 12:00:00', 'SEA', 'ATL'),
          predict_delay('30/1/2018 15:00:00', 'SEA', 'ATL'),
          predict_delay('30/1/2018 18:00:00', 'SEA', 'ATL'),
          predict_delay('30/1/2018 21:00:00', 'SEA', 'ATL'))
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))


# In[ ]:




