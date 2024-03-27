import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("./dataset.csv")
# print(data.head())

# Let’s have a look at whether this dataset contains any null values or not
data.isnull().sum()

# let’s have a look at all the languages present in this dataset
data["language"].value_counts()

#let’s split the data into training and test sets
text = np.array(data["Text"])
lang = np.array(data["language"])
cv = CountVectorizer()
X = cv.fit_transform(text)
X_train, X_test, Y_train, Y_test = train_test_split(X, lang,test_size=0.33, random_state=42)

# we  will be using the Multinomial Naïve Bayes algorithm to train the language detection model 
model = MultinomialNB()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

# let’s use this model to detect the language of a text by taking a user input
user = input("Enter your Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
