# Masoud Pourghavam
# Student Number: 810601044
# Course: Artificial Intelligence
# University of Tehran
# Homework 3


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# Disable all warnings
warnings.filterwarnings("ignore")

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('musics.csv')

# Drop the 1st, 2nd, and 5th columns from the DataFrame
df.drop(df.columns[[0, 1, 4]], axis=1, inplace=True)

# Define the mapping of ranges to replacement values
popularity_mapping = {range(0, 21): 1, range(21, 41): 2, range(41, 61): 3, range(61, 81): 4, range(81, 101): 5}

# Replace the values in the "popularity" column based on the mapping
df['popularity'] = df['popularity'].map({value: replacement for key, replacement in popularity_mapping.items() for value in key})

# Replace "true" with 0 and "false" with 1 in the "explicit" column
df['explicit'].replace({'true': 0, 'false': 1}, inplace=True)

# Create a new column "new_genre" by combining genres into a single string
df['new_genre'] = df['genre'].apply(lambda x: ', '.join(x.split(',')))

# Split the genres in the "genre" column and create a list of genre lists
df['genre'] = df['genre'].str.split(',')

# Perform one-hot encoding on the genre column
genres = df['genre'].explode()
one_hot_encoded = pd.get_dummies(genres, prefix='genre')

# Group the one-hot encoded genres by index and sum the values
one_hot_encoded = one_hot_encoded.groupby(level=0).sum()

# Concatenate the original DataFrame with the one-hot encoded genres
df = pd.concat([df, one_hot_encoded], axis=1)

# Put the split genres list in the new "new_genre" column
df['new_genre'] = df['genre'].apply(lambda x: ', '.join(x))



# Select columns for plotting
columns_to_plot = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
                   'mode', 'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo']

# Set the figure size and layout
plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)

# Loop through each column and plot the data distribution
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[column], kde=True)
    plt.title(column)
    plt.xlabel('')
    plt.ylabel('Count')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# Set the figure size
plt.figure(figsize=(8, 6))

# Define the colors for each popularity value
colors = {1: 'blue', 2: 'red', 3: 'yellow', 4: 'purple', 5: 'brown'}

# Filter the DataFrame for the specified popularity values
popularity_values = [1, 2, 3, 4, 5]
df_filtered = df[df['popularity'].isin(popularity_values)]

# Create the histogram plot for each popularity value with the assigned color
for value, color in colors.items():
    sns.histplot(data=df_filtered[df_filtered['popularity'] == value], x='popularity', bins=20,
                 kde=True, color=color, label=f'Popularity {value}')

# Set the plot title and labels
plt.title('Distribution of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Count')

# Add a legend
plt.legend()

# Show the plot
plt.show()



# Select the columns for visualization
columns_to_visualize = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
                        'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']

# Create a scatter plot matrix
sns.set(style='ticks')
sns.pairplot(df[columns_to_visualize])

# Show the plot
plt.show()



# Select the columns to normalize
columns_to_normalize = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
                        'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']

# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Normalize the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Show the normalized DataFrame
print(df.head())

# Select the columns of interest
columns_of_interest = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
                       'mode', 'speechiness', 'acousticness', 'instrumentalness',
                       'liveness', 'valence', 'tempo', 'popularity']

# Create a correlation matrix
corr_matrix = df[columns_of_interest].corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)

# Show the plots
plt.show()

# Split the data into training and temporary sets
train_temp, test_val = train_test_split(df, test_size=0.2, random_state=42)

# Split the temporary set into testing and validation sets (50% of the temporary set)
test, val = train_test_split(test_val, test_size=0.5, random_state=42)

# Verify the sizes of the splits
print(f"Training set size: {train_temp.shape[0]}")
print(f"Testing set size: {test.shape[0]}")
print(f"Validation set size: {val.shape[0]}")

print ("############################################################")

# Define the features (X) and the target variable (y)
features = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo']
target = 'popularity'

# Split the data into training and temporary data
train_data, temp_data, train_target, temp_target = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Split the temporary data into testing and validation data
test_data, val_data, test_target, val_target = train_test_split(temp_data, temp_target, test_size=0.5, random_state=42)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(train_data, train_target)

# Predict the popularity for the testing data
predictions = classifier.predict(test_data)

# Plot the decision tree classifier
plt.figure(figsize=(15, 10))
tree.plot_tree(classifier, feature_names=features, class_names=['1', '2', '3', '4', '5'], filled=True)
plt.show()

# Create a confusion matrix
cm = confusion_matrix(test_target, predictions)

# Calculate the accuracy score
accuracy = accuracy_score(test_target, predictions)

# Calculate the precision score
precision = precision_score(test_target, predictions, average='weighted')

# Calculate the recall score
recall = recall_score(test_target, predictions, average='weighted')

# Calculate the F1-score
f1 = f1_score(test_target, predictions, average='weighted')

# Calculate the error rate
error_rate = 1 - accuracy


# Print the accuracy, precision, recall, F1-score, error rate, and specificity scores
print(f"Accuracy of decision tree: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Error rate: {error_rate}")


# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()


print ("############################################################")


# Create a random forest classifier
classifier = RandomForestClassifier()

# Train the classifier on the training data
classifier.fit(train_data, train_target)

# Predict the popularity for the testing data
predictions = classifier.predict(test_data)

# Create a confusion matrix
cm = confusion_matrix(test_target, predictions)

# Calculate the accuracy score
accuracy = accuracy_score(test_target, predictions)

# Calculate the precision score
precision = precision_score(test_target, predictions, average='weighted')

# Calculate the recall score
recall = recall_score(test_target, predictions, average='weighted')

# Calculate the F1-score
f1 = f1_score(test_target, predictions, average='weighted')

# Calculate the error rate
error_rate = 1 - accuracy


# Print the accuracy, precision, recall, F1-score, error rate, and specificity scores
print(f"Accuracy of random forest: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Error rate: {error_rate}")

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()


print ("############################################################")


# Create a KNN classifier
classifier = KNeighborsClassifier()

# Train the classifier on the training data
classifier.fit(train_data, train_target)

# Predict the popularity for the testing data
predictions = classifier.predict(test_data)

# Create a confusion matrix
cm = confusion_matrix(test_target, predictions)

# Calculate the accuracy score
accuracy = accuracy_score(test_target, predictions)

# Calculate the precision score
precision = precision_score(test_target, predictions, average='weighted')

# Calculate the recall score
recall = recall_score(test_target, predictions, average='weighted')

# Calculate the F1-score
f1 = f1_score(test_target, predictions, average='weighted')

# Calculate the error rate
error_rate = 1 - accuracy

# Print the accuracy, precision, recall, F1-score, error rate, and specificity scores
print(f"Accuracy of KNN: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Error rate: {error_rate}")

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()


print ("############################################################")

# Create an SVM classifier
classifier = SVC()

# Train the classifier on the training data
classifier.fit(train_data, train_target)

# Predict the popularity for the testing data
predictions = classifier.predict(test_data)

# Create a confusion matrix
cm = confusion_matrix(test_target, predictions)

# Calculate the accuracy score
accuracy = accuracy_score(test_target, predictions)

# Calculate the precision score
precision = precision_score(test_target, predictions, average='weighted')

# Calculate the recall score
recall = recall_score(test_target, predictions, average='weighted')

# Calculate the F1-score
f1 = f1_score(test_target, predictions, average='weighted')

# Calculate the error rate
error_rate = 1 - accuracy


# Print the accuracy, precision, recall, F1-score, error rate, and specificity scores
print(f"Accuracy of SVM: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Error rate: {error_rate}")


# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

print ("############################################################")


# Save the modified DataFrame back to a new CSV file
df.to_csv('updated_dataset.csv', index=False)

df = pd.read_csv("musics.csv")

# Split the genres and create a list of all genres
all_genres = df["genre"].str.split(",").explode().str.strip()

# Find the most frequent genre
most_frequent_genre = all_genres.mode().values[0]

# Find the least frequent genre
least_frequent_genre = all_genres.value_counts().idxmin()

print("Most frequent genre:", most_frequent_genre)
print("Least frequent genre:", least_frequent_genre)

print ("############################################################")

# Split the genres and create a list of all genres
all_genres = df["genre"].str.split(",").explode().str.strip()

# Calculate the genre frequencies
genre_counts = all_genres.value_counts()

# Set up the figure and axes
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the histogram of genre frequencies
genre_counts.plot(kind='bar', ax=ax)

# Set the axis labels and title
ax.set_xlabel("Genre")
ax.set_ylabel("Frequency")
plt.title("Genre Frequencies")

# Show the most frequent and least frequent genres as annotations
most_frequent_genre = genre_counts.idxmax()
least_frequent_genre = genre_counts.idxmin()
ax.annotate(f"Most Frequent: {most_frequent_genre}", xy=(0.5, 0.8), xycoords='axes fraction', ha='center')
ax.annotate(f"Least Frequent: {least_frequent_genre}", xy=(0.5, 0.7), xycoords='axes fraction', ha='center')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Display the plot
plt.tight_layout()
plt.show()


print ("############################################################")

# Count the number of genres for each song
genre_counts = df["genre"].str.split(",").apply(lambda x: len(x))

# Find the maximum number of genres
max_num_genres = genre_counts.max()

# Get the songs with the most number of genres
songs_with_max_genres = df[genre_counts == max_num_genres]

print("Most number of genres for a song:", max_num_genres)
print("Songs with the most number of genres:")
print(songs_with_max_genres)


print ("############################################################")


# Calculate the probability of a song being placed in other genres using Bayes' rule
target_genre = "R&B"  # Replace with the target genre you want to calculate the probability for

target_genre_count = all_genres.value_counts()[target_genre]
total_songs = len(df)

probability_other_genres = (total_songs - target_genre_count) / total_songs

print("Probability of a song being placed in other genres:", probability_other_genres)

# Get the unique genres from the 'genre' column
genres = df['genre'].unique()

# Calculate the probability of each genre
total_songs = len(df)
genre_probabilities = {}

for genre in genres:
    genre_count = len(df[df['genre'] == genre])
    genre_probability = genre_count / total_songs
    genre_probabilities[genre] = genre_probability

# Calculate the probability of each song belonging to each genre
results = []

for index, row in df.iterrows():
    song_genres = row['genre'].split(', ')
    song_probabilities = {}

    for genre in genres:
        if genre in song_genres:
            genre_probability = genre_probabilities[genre]
        else:
            genre_probability = (1 - genre_probabilities[genre]) / (len(genres) - len(song_genres))
        song_probabilities[genre] = genre_probability

    results.append(song_probabilities)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('song_probabilities.csv', index=False)


print ("############################################################")


# Load the dataset
df = pd.read_csv("updated_dataset.csv")

# Split the data into features (X) and the target variable (y)
X = df.drop("genre", axis=1)  # Features
y = df["genre"]               # Target variable

# Perform one-hot encoding on the genre column
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Split the data into training set, testing set, and validation set
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier on the training set
clf.fit(X_train, y_train)

# Predict the genre of songs in the testing set
y_pred = clf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate the precision score
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate the recall score
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate the F1-score
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate the error rate
error_rate = 1 - accuracy

# Print the accuracy, precision, recall, F1-score, error rate, and specificity scores
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Error rate: {error_rate}")
print(f"Accuracy: {accuracy}")

# Show the plot
plt.show()

print ("############################################################")









