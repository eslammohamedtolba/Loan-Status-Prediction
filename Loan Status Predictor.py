# import required modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# loading dataset 
Loan_status_data = pd.read_csv("/content/train.csv")
# show dataset
Loan_status_data.head()
# show the number of rows and columns
Loan_status_data.shape
# find statistical info about the dataset
Loan_status_data.describe()


# data cleaning
Loan_status_data.isnull().sum() # check if there is any None values in the dataset in eadh column
# (1) replace the none values with its column mean => return array
# strategy = SimpleImputer(missing_values = np.nan,strategy= 'mean')
# Loan_status_data = strategy.fit_transform(Loan_status_data)
# (2) drop any row that contains at least one none value => return dataframe
Loan_status_data= Loan_status_data.dropna()



#(1) create labelencoder to convert any string column to numerical column
le = LabelEncoder()
# find all columns datatypes 
datatypes = Loan_status_data.dtypes
# convert any textual column to numerical column
for colIndex in range(len(datatypes)):
    if datatypes[colIndex]=='object':
        Loan_status_data.iloc[:,colIndex] = le.fit_transform(Loan_status_data.iloc[:,colIndex])
Loan_status_data.head()
#(2) knowing the different values in each column and its repetition
# Loan_status_data['column_name'].value_counts()
# Loan_status_data.replace({'column_name':{'value1':replacedvalue1,'value2':replacedvalue2}},inplace=True)


# show the dataset after applying cleaing and labeling data
Loan_status_data.head()


# find the relation between the output and each feature in the input
Loan_status_data.groupby('Loan_Status').mean()
# find correlation between various features in the dataset
correlation = Loan_status_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap = 'Blues')
# find the relation between the two columns Education && Loan_Status
sns.countplot(x = 'Education',hue = 'Loan_Status',data=Loan_status_data)
# find the relation between the two columns Married && Loan_Status
sns.countplot(x = 'Married',hue = 'Loan_Status',data=Loan_status_data)


# split data into input and label data
X = Loan_status_data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = Loan_status_data['Loan_Status']
print(X)
print(Y)


# scale the input data to take common range
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)


# split the data to train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
# show the shape of train and test data
print(x_train.shape,x_test.shape,X.shape)
print(y_train.shape,y_test.shape,Y.shape)



# create model and train it
SVCModel = svm.SVC()
SVCModel.fit(x_train,y_train)
# make the model predict test and train data input
predicted_train_data = SVCModel.predict(x_train)
predicted_test_data = SVCModel.predict(x_test)
# find accuracy score for training and test data predictions
accuracy_train_data_prediction = accuracy_score(predicted_train_data,y_train)
accuracy_test_data_prediction = accuracy_score(predicted_test_data,y_test)
print(accuracy_train_data_prediction,accuracy_test_data_prediction)
# show predicted values and actual values for train and test data 
for value in zip(predicted_train_data,y_train):
    print(value)
for value in zip(predicted_test_data,y_test):
    print(value)


# making a predictive system
input_data = (0,0,0,1,0,6000,0,141,360,1,1) # user input data
# convert data input into 1D numpy array
input_data_numpyarray = np.array(input_data)
# convert 1D numpy array Data into 2D numpy array data
input_data_2d_numpyarray = input_data_numpyarray.reshape(1,-1)
if SVCModel.predict(input_data_2d_numpyarray)[0]==1:
    print("Yes")
else:
    print("No")



