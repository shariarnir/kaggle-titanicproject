#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Hello Jupyter!")


# In[3]:


#Definition of radius in km
r = 192500
#import radians function of math package
from math import radians
dist = r * radians(12)
print(dist)


# In[4]:


import pandas as pd
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')


# In[5]:


import pandas as pd
train = pd.read_csv('datasets/titanic/train.csv')
test = pd.read_csv('datasets/titanic/test.csv')


# In[6]:


import pandas as pd
train = pd.read_csv('C:/Users/shariar_nir/Documents/datasets/titanic/train.csv')
test = pd.read_csv('C:/Users/shariar_nir/Documents/datasets/titanic/test.csv')


# In[7]:


train.head(5)


# In[9]:


train.shape


# In[10]:


test.shape


# In[11]:


train.info()


# In[12]:


test.info()


# In[13]:


train.isnull().sum()


# In[14]:


test.isnull().sum()


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[22]:


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked = True, figsize = (10,5))
    


# In[23]:


bar_chart('Sex')


# In[24]:


bar_chart('Pclass')


# In[26]:


bar_chart('SibSp')


# In[27]:


bar_chart('Parch')


# In[28]:


bar_chart('Embarked')


# In[31]:


train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand = False)
    


# In[32]:


train['Title'].value_counts()


# In[33]:


test['Title'].value_counts()


# In[40]:


title_mapping = {"Mr": 0,"Miss": 1,"Mrs": 2,
                 "Master": 3,"Dr": 3,"Rev": 3,
                 "Col": 3,"Major": 3,"Mlle": 3,
                 "Countess": 3,"Ms": 3,"Lady": 3,
                 "Jonkheer": 3,"Don": 3,"Dona": 3,
                 "Mme": 3,"Capt": 3,"Sir": 3}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)


# In[41]:


train.head()


# In[42]:


test.head()


# In[43]:


bar_chart('Title')


# In[44]:


#delete unnecessary feature from dataset
train.drop('Name',axis = 1, inplace = True)
test.drop('Name',axis = 1, inplace = True)


# In[45]:


train.head()


# In[46]:


test.head()


# In[47]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[48]:


bar_chart('Sex')


# In[49]:


train.head()


# In[54]:


#fill missing age with median age for each title(Mr,Mrs,Miss,Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace =True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace =True)


# In[55]:


train.groupby("Title")["Age"].transform("median")
train.head()


# In[57]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.show()


# In[60]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.xlim(0,20)


# In[61]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.xlim(20,30)


# In[62]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.xlim(30,40)


# In[63]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.xlim(40,60)


# In[64]:


train.info()


# In[65]:


test.info()


# In[97]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16,'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62,'Age'] = 4


# In[99]:


train.head()


# In[100]:


bar_chart('Age')


# In[102]:


Pclass1 = train[ train[ 'Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[ train[ 'Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[ train[ 'Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = [ '1st Class','2nd Class','3rd Class']
df.plot(kind = 'bar',stacked = True, figsize = (10,5))


# In[103]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[104]:


train.head()


# In[105]:


embarked_mapping = {"S":0, "C": 1,"Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[106]:


#fill missing fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)
train.head(5)


# In[107]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()
plt.show()


# In[108]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0,20)


# In[109]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0,30)


# In[110]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17,'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100,'Fare'] = 3


# In[111]:


train.head()


# In[112]:


train.Cabin.value_counts()


# In[113]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[114]:


Pclass1 = train[ train[ 'Pclass'] == 1]['Cabin'].value_counts()
Pclass2 = train[ train[ 'Pclass'] == 2]['Cabin'].value_counts()
Pclass3 = train[ train[ 'Pclass'] == 3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = [ '1st Class','2nd Class','3rd Class']
df.plot(kind = 'bar',stacked = True, figsize = (10,5))


# In[115]:


cabin_mapping = {"A":0, "B": 0.4,"C": 0.8,"D": 1.2, "E": 1.6,"F": 2,"G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[116]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace = True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace = True)


# In[117]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[118]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'FamilySize',shade = True)
facet.set(xlim = (0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[119]:


family_mapping = {1:0, 2: 0.4,3: 0.8,4: 1.2, 5: 1.6,6: 2,7: 2.4, 8: 2.8,9: 3.2,10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[120]:


train.head()


# In[121]:


train.head()


# In[122]:


feature_drop = ['Ticket','SibSp','Parch']
train = train.drop(feature_drop,axis = 1)
test = test.drop(feature_drop,axis = 1)
train = train.drop(['PassengerId'],axis = 1)


# In[123]:


train_data = train.drop('Survived', axis = 1)
target = train['Survived']
train_data.shape, target.shape


# In[125]:


train_data.head()


# In[126]:


# Importing classifier Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[127]:


train.info()


# In[128]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10,shuffle = True,random_state = 0)


# In[130]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[131]:


#decision tree score
round(np.mean(score)*100,2)


# In[132]:


clf = RandomForestClassifier(n_estimators = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[133]:


#Random forest score
round(np.mean(score)*100,2)


# In[134]:


clf = RandomForestClassifier(n_estimators = 13)
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis = 1).copy()
prediction = clf.predict(test_data)


# In[136]:


submission = pd.DataFrame({
    "PassengerId" : test["PassengerId"],"Survived" : prediction
})
submission.to_csv('submission.csv',index = False)


# In[140]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:




