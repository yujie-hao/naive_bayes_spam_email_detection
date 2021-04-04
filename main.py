import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

data_dir = "./input/"

df = pd.read_csv(data_dir + 'spam.csv', encoding='latin-1')

print(df.head())

print(df.v2.head(1))

print(df.v1.head())

print(df.shape)

data_train, data_test, labels_train, labels_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state=0)

print(data_train.shape)

print(data_test.shape)

vectorizer = CountVectorizer()
data_train_demo = ["We are good students", "You are good student"]
data_train_count_demo = vectorizer.fit_transform(data_train_demo)

print(vectorizer.vocabulary_)
print(data_train_count_demo.toarray())

data_train_count = vectorizer.fit_transform(data_train)
data_test_count = vectorizer.transform(data_test)
occurrence = data_train_count.toarray().sum(axis=0)
plt.plot(occurrence)
plt.show()

word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrence': occurrence})
word_freq_df_sort = word_freq_df.sort_values(by=['occurrence'], ascending=False)

print(word_freq_df_sort.head())

classifier = MultinomialNB()
classifier.fit(data_train_count, labels_train)
MultinomialNB()
predictions = classifier.predict(data_test_count)
print(predictions)

print(accuracy_score(labels_test, predictions))

print(classification_report(labels_test, predictions))
print(confusion_matrix(labels_test, predictions))

data_content = df.v2
data_label = df.v1
vect = CountVectorizer()
data_count = vect.fit_transform(data_content)
cross_val = cross_val_score(classifier, data_count, data_label, cv=20, scoring='accuracy')
print(cross_val)
print(np.mean(cross_val))