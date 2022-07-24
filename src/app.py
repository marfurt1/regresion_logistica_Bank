import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV

#Let's load the data and take a first look to the first rows.


#load data
url = 'https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv'
data = pd.read_csv(url, sep=";")


def replace_with_frequent(df,col):
    frequent = df[col].value_counts().idxmax()
    print("The most frequent value is:", frequent)
    df[col].replace('unknown', frequent , inplace = True)
    print("Replacing unknown values with the most frequent value:", frequent)

#Replacing unknown values in categorical features.
print('Job:')
replace_with_frequent(data, "job")
print('-'*50)
print('Marital:')
replace_with_frequent(data, "marital")
print('-'*50)
print('Education:')
replace_with_frequent(data, "education")
print('-'*50)
print('Default:')
replace_with_frequent(data, "default")
print('-'*50)
print('Housing:')
replace_with_frequent(data, "housing")
print('-'*50)
print('Loan:')
replace_with_frequent(data, "loan")

#add a new column next to the age column for age groups.

age_groups = pd.cut(data['age'],bins=[10,20,30,40,50,60,70,80,90,100],
                    labels=['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-100'])

#inserting the new column
data.insert(1,'age_group',age_groups)

#dropping age column
data.drop('age',axis=1,inplace=True)

## grouping education categories 'basic.9y','basic.6y','basic4y' into 'middle_school'**

lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    data.loc[data['education'] == i, 'education'] = "middle.school"

#remove duplicated  keeping the most recent one.

duplicated_data=data[data.duplicated(keep="last")]
data=data.drop_duplicates()

##Converting target variable into binary**
def target_to_binary(y):
    y.replace({"yes":1,"no":0},inplace=True)
target_to_binary(data['y'])

## Encoding ordinal features**

encoder = LabelEncoder()
data['age_group'] = encoder.fit_transform(data['age_group'])
data['education'] = encoder.fit_transform(data['education'])

## Encoding categorical features that are not ordinal**
data = pd.get_dummies(data, columns = ['job', 'marital', 'default','housing', 'loan', 'contact', 'poutcome'])

## Encoding month and day of the week**
month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12,'ene':1, 'feb':2}
data['month']= data['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6, 'sun':1, 'sat':7}
data['day_of_week']= data['day_of_week'].map(day_dict)




