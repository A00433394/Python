import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy
import pickle
from sklearn.svm import LinearSVC

print("Loading data....\n")
file = pd.read_csv("imdb_labelled.txt", sep= "\t", names=['txt', 'liked'])

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

print("Vectorizing data \n")
# total Observation and Unique words
y = file.liked
x = vectorizer.fit_transform(file.txt)

print("Splitting into test and train sets \n")
# split of train and test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, test_size=0.2, train_size=0.8)


print(" ---------------------- NAIVE BAYES------------------------")

# Loading Baye's Model
classifier_f = open("bayes.pickle", "rb")
classify = pickle.load(classifier_f)
classifier_f.close()


# Accuracy
acc = roc_auc_score(Y_test, classify.predict_proba(X_test)[:, 1])
print("Accuracy for Naive Baye's Classifier Model is ", acc)
print("")

print("----------------------- SVM --------------------------------")

# Loading SVM Model
classifier_svm = open("svm.pickle", "rb")
modelsvc= pickle.load(classifier_svm)
classifier_svm.close()

# Accuracy
acc1 = str(modelsvc.score(X_test, Y_test))
print("Accuracy for SVC Classifier Model is ", acc1)

print("")
print("----------------------- RANDOM FOREST -----------------------")

# Loading rf Model
classifier_rf = open("rf.pickle", "rb")
modelRf = pickle.load(classifier_rf)
classifier_rf.close()

# Accuracy
acc2 = roc_auc_score(Y_test, modelRf.predict_proba(X_test)[:, 1])
print("Accuracy for Test Random Forest Classifier Model is", acc2)

print("  ")
print("---------------------------------")

# checking on an input data
movie_review = numpy.array(["I Love the movie"])
movie_review_vector = vectorizer.transform(movie_review)
print("The Scentiment from Naive Baye's is", classify.predict(movie_review_vector))
print("The Scentiment from SVC is", modelsvc.predict(movie_review_vector))
print("The Scentiment from RF is", modelRf.predict(movie_review_vector))

