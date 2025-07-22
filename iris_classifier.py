import numpy as np     #numerical python -> math operation
import pandas as pd     #data handling
import matplotlib.pyplot as plt  #create graph/charts
import seaborn as sns  #on top of matplotlib, beautiful statistical plots

# from sklearn.datasets import load_iris       #load iris dataset    #am loading dataset manually below
from sklearn.model_selection import train_test_split  #for splitting data
from sklearn.linear_model import LogisticRegression   #ML algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#manually loading iris-dataset
df=pd.read_csv('iris.csv', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df['species'] = df['species'].str.strip()

print(df.head(150-145))   # table like date stored in df, df.head() means top 5 tuples
# data loaded successfully -> now visualize data

print('dataset Info :\n')
print(df.info())

print('summary statistics :\n')
print(df.describe())

print('species count :\n')
print(df['species'].value_counts())  #count of each species : setosa, virginica, versicolor

#step 2: pairplot
sns.pairplot(df, hue='species')
plt.suptitle("Iris Dataset Pair Plot", y=1.02)
#plt.show()    # seaborn tell how data is distrubuted  -> now split the dataset and train a Logistic regression model

# features x and target y
x=df.drop('species', axis=1)
y=df['species']

# #split in training data & testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# #initialise and train model
model=LogisticRegression(max_iter=200)
model.fit(x_train, y_train)   #train the model
 
#make prediction
y_pred = model.predict(x_test) #predict label for test

#evaluate the model
print("Model accuracy : \n", 100*accuracy_score(y_test, y_pred),"%")
print("classification report : \n", classification_report(y_test, y_pred))

features = []
features.append(float(input("Enter sepal length: ")))
features.append(float(input("Enter sepal width: ")))
features.append(float(input("Enter petal length: ")))
features.append(float(input("Enter petal width: ")))


custom_input = pd.DataFrame([features], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
predicted_class = model.predict(custom_input)

print("Predicted Iris Class is =>", predicted_class[0])
