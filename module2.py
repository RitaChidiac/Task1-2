
from cgi import test
from pickle import NONE
from sre_parse import CATEGORIES
from pandas.tseries.offsets import Second
import prophet
from prophet.plot import plot_components_plotly, plot_plotly

import wget
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import Input, Output, dcc, html
import plotly.graph_objs as go
import numpy as np
import datetime


   


df=pd.read_csv(r"Titanic-Dataset.csv")
print(df)
df=df.drop_duplicates()
print(df)
df.columns
df.head()
df.tail()
print(df.tail())
print(df.isnull().sum())
df.Cabin=df.Cabin.fillna("unknown")
print(df.isnull().sum())
print(df.shape)#nb rows*nb col
print(df.dtypes)#each column data type
print(df)
print(df["Survived"].value_counts())
print("Total numbers of passengers ",len(df))
print("Total number of survival",len(df[df["Survived"]==1]))
print("Total number of passenger who did not survive",len(df[df["Survived"]==0]))
print(df["Sex"].value_counts())
import numpy as np
print("% of female who survived",100*np.mean(df["Survived"][df["Sex"]=="female"]))
print("% of female who survived",100*np.mean(df["Survived"][df["Sex"]=="male"]))
print("% of passengers who survived in first class",100*np.mean(df["Survived"][df["Pclass"]==1]))
print("% of passengers who survived in third class",100*np.mean(df["Survived"][df["Pclass"]==3]))    
print(df[["Survived","Pclass"]].groupby(["Pclass"]).mean())
print(df.shape)
df.info()
print(df["Age"].value_counts())
print(df["Cabin"])
df2=df.copy()
print(df2)
df2["Sex"]=df["Sex"].apply(lambda x: 1 if x=="male" else 0)
print(df2["Sex"])
df2=df.copy()
print(df2.isnull().sum())
df2["Age"]=df2["Age"].fillna(np.mean(df2["Age"]))
print(df2.isnull().sum())
embark=df2["Embarked"].dropna()
print(df2['Embarked'].mode())
df2['Embarked'].fillna(df2['Embarked'].mode()[0],inplace=True)

print(df2.isnull().sum())
print(df.describe())


#Task2
import matplotlib.pyplot as plt
# Count the number of passengers in each class
titanic=df2['Pclass'].value_counts()


print(titanic)

#Bar Chart: Distribution of Passengers by Class
plt.figure(figsize=(8, 6))
titanic.plot(kind='bar', color='skyblue')
plt.title('Distribution of Passengers by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0)
plt.show()


#Scatter Plot: Relationship between Age and Fare

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df2['Age'], df2['Fare'], alpha=0.5)
plt.title('Scatter Plot: Age vs. Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# Create a scatter plot with different colors for survival status
plt.figure(figsize=(10, 6))
plt.scatter(df2[df2['Survived'] == 1]['Age'],
            df2[df2['Survived'] == 1]['Fare'],
            alpha=0.5, color='green', label='Survived')
plt.scatter(df2[df2['Survived'] == 0]['Age'],
            df2[df2['Survived'] == 0]['Fare'],
            alpha=0.5, color='red', label='Did Not Survive')
plt.title('Scatter Plot: Age vs. Fare (Survival)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.show()
import numpy as np
import seaborn as sns
sns.countplot(x='Survived',data=df2)
plt.title("Not survived and survived")
plt.xlabel("survival")
plt.ylabel("count")
plt.show()

first_class_count=(df2['Pclass']==1).sum()
print("First class:",first_class_count)
Second_class_count=(df2['Pclass']==2).sum()
print("Second class:",Second_class_count)
third_class_count=(df2['Pclass']==3).sum()
print("Third class:",third_class_count)

labels=['First class','Second class','Third class']
sizes=[first_class_count,Second_class_count,third_class_count]
plt.pie(sizes,labels=labels,autopct='%1.2f%%')
plt.axis("equal")
plt.title('class distibution')
plt.show()


plt.figure(figsize=(16,8))
sns.countplot(x=df2["Pclass"],hue=df2["Survived"])
plt.title("survived via class")
plt.xlabel("classes")
plt.ylabel("count")
plt.show()


plt.figure(figsize=(16,8))
axs=sns.kdeplot(df2.Age[df2.Survived==0],label="died")
axs=sns.kdeplot(df2.Age[df2.Survived==1],label="survived")
plt.title("survived via Age")
plt.xlabel("Age")
plt.ylabel("Normalized count")
plt.show()


male_dead=((df2['Sex']=='male')&(df2['Survived']==0)).sum()

male_survived=((df2['Sex']=='male')&(df2['Survived']==1)).sum()

female_dead=((df2['Sex']=='female')&(df2['Survived']==0)).sum()

female_survied=((df2['Sex']=='female')&(df2['Survived']==1)).sum()

m_data=(male_dead,male_survived)
f_data=(female_dead,female_survied)

plt.figure(figsize=(16,8))
p1=plt.bar(np.arange(2),(m_data),width=0.3)
p2=plt.bar(np.arange(2),(f_data),bottom=m_data,width=0.3)
plt.xticks(np.arange(2),["Men","women"])
plt.legend((p1[0], p2[0]), ("Died", "Survived"))
plt.title("Women survival vs. men survival")
plt.show()
 



plt.figure(figsize=(16,8))
sns.countplot(x=df2["Embarked"],hue=df2["Survived"])
plt.title("Boarding station count survival")
plt.xlabel("Port")
plt.ylabel("count")
plt.show()







