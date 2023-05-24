# Masoud Pourghavam
# Student Number: 810601044
# Course: Artificial Intelligence
# University of Tehran
# Homework 3

import warnings
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Disable all warnings
warnings.filterwarnings("ignore")

# dataframes creation for both training and testing datasets
iris_df = pd.read_csv('Iris.csv')

sns.scatterplot( x = 'Sepal_Length', y = 'Sepal_Width', hue = 'Class', data = iris_df)
plt.show()

sns.scatterplot(x = 'Petal_Length', y = 'Petal_Width', hue = 'Class', data = iris_df)
plt.show()

# Let's try the Seaborn pairplot
sns.pairplot(iris_df, hue = 'Class')
plt.show()

# Let's drop the ID and class (target label) columns
X = iris_df.drop(['Class'],axis=1)
y = iris_df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


print("####################################################"
      ""
      "")

# Calculate Jaccard score for each class
jaccard_scores = jaccard_score(y_test, y_predict, average=None)

# Print Jaccard score for each class
print("Jaccard score for Iris-setosa: ", jaccard_scores[0])
print("Jaccard score for Iris-versicolor: ", jaccard_scores[1])
print("Jaccard score for Iris-virginica: ", jaccard_scores[2])

# Calculate macro average of Jaccard score
macro_avg_jaccard_score = np.mean(jaccard_scores)
print("Macro avg accuracy of Jaccard score: ", macro_avg_jaccard_score)

print("####################################################"
      ""
      "")
# Calculate recall scores
recall_scores = recall_score(y_test, y_predict, average=None)
print("Recall score for Iris-setosa: ", recall_scores[0])
print("Recall score for Iris-versicolor: ", recall_scores[1])
print("Recall score for Iris-virginica: ", recall_scores[2])

# Calculate macro average of recall scores
macro_avg_recall_score = np.mean(recall_scores)
print("Macro avg accuracy of Recall score: ", macro_avg_recall_score)

print("####################################################"
      ""
      "")
# Calculate precision scores
precision_scores = precision_score(y_test, y_predict, average=None)
print("Precision score for Iris-setosa: ", precision_scores[0])
print("Precision score for Iris-versicolor: ", precision_scores[1])
print("Precision score for Iris-virginica: ", precision_scores[2])

# Calculate macro average of precision scores
macro_avg_precision_score = np.mean(precision_scores)
print("Macro avg accuracy of Precision score: ", macro_avg_precision_score)

print("####################################################"
      ""
      "")

# Calculate classification report
cr = classification_report(y_test, y_predict, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], output_dict=True)

# Calculate accuracy score for each class
setosa_acc = (cr['Iris-setosa']['precision'] + cr['Iris-setosa']['recall']) / 2
versicolor_acc = (cr['Iris-versicolor']['precision'] + cr['Iris-versicolor']['recall']) / 2
virginica_acc = (cr['Iris-virginica']['precision'] + cr['Iris-virginica']['recall']) / 2

# Calculate macro average
macro_avg = (setosa_acc + versicolor_acc + virginica_acc) / 3

# Print results
print("Accuracy score for Iris-setosa: ", setosa_acc)
print("Accuracy score for Iris-versicolor: ", versicolor_acc)
print("Accuracy score for Iris-virginica: ", virginica_acc)
print("Macro average of accuracy scores: ", macro_avg)

print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")

print("##### AFTER NORMALIZATION #####")

# Perform standard normalization on the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()

# Calculate Jaccard score for each class
jaccard_scores = jaccard_score(y_test, y_predict, average=None)

# Print Jaccard score for each class
print("Jaccard score for Iris-setosa: ", jaccard_scores[0])
print("Jaccard score for Iris-versicolor: ", jaccard_scores[1])
print("Jaccard score for Iris-virginica: ", jaccard_scores[2])

# Calculate macro average of Jaccard score
macro_avg_jaccard_score = np.mean(jaccard_scores)
print("Macro avg accuracy of Jaccard score: ", macro_avg_jaccard_score)

print("####################################################"
      ""
      "")
# Calculate recall scores
recall_scores = recall_score(y_test, y_predict, average=None)
print("Recall score for Iris-setosa: ", recall_scores[0])
print("Recall score for Iris-versicolor: ", recall_scores[1])
print("Recall score for Iris-virginica: ", recall_scores[2])

# Calculate macro average of recall scores
macro_avg_recall_score = np.mean(recall_scores)
print("Macro avg accuracy of Recall score: ", macro_avg_recall_score)

print("####################################################"
      ""
      "")
# Calculate precision scores
precision_scores = precision_score(y_test, y_predict, average=None)
print("Precision score for Iris-setosa: ", precision_scores[0])
print("Precision score for Iris-versicolor: ", precision_scores[1])
print("Precision score for Iris-virginica: ", precision_scores[2])

# Calculate macro average of precision scores
macro_avg_precision_score = np.mean(precision_scores)
print("Macro avg accuracy of Precision score: ", macro_avg_precision_score)

print("####################################################"
      ""
      "")

# Calculate classification report
cr = classification_report(y_test, y_predict, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], output_dict=True)

# Calculate accuracy score for each class
setosa_acc = (cr['Iris-setosa']['precision'] + cr['Iris-setosa']['recall']) / 2
versicolor_acc = (cr['Iris-versicolor']['precision'] + cr['Iris-versicolor']['recall']) / 2
virginica_acc = (cr['Iris-virginica']['precision'] + cr['Iris-virginica']['recall']) / 2

# Calculate macro average
macro_avg = (setosa_acc + versicolor_acc + virginica_acc) / 3

# Print results
print("Accuracy score for Iris-setosa: ", setosa_acc)
print("Accuracy score for Iris-versicolor: ", versicolor_acc)
print("Accuracy score for Iris-virginica: ", virginica_acc)
print("Macro average of accuracy scores: ", macro_avg)