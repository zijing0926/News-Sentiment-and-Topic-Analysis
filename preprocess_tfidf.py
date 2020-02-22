# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:53:31 2020

@author: zzhu1
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_excel('news_sample.xlsx')
df.dropna(inplace=True)
##check tfidf for each tokenized words
docs=df['txt']

###preprocessing txt
import re
import nltk
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker   
from sklearn.feature_extraction.text import CountVectorizer

##define tokenize, stemming and stopwords
spell = SpellChecker() 
porter_stemmer = PorterStemmer()

###stopwords
with open('StopWords_GenericLong.txt', 'r') as f:
    x_gl = f.readlines()
with open('StopWords_Names.txt', 'r') as f:
    x_n = f.readlines()
with open('StopWords_DatesandNumbers.txt', 'r') as f:
    x_d = f.readlines()
with open('StopWords_Geographic.txt', 'r') as f:
    x_g = f.readlines()
stopwords = nltk.corpus.stopwords.words('english')
[stopwords.append(x) for x in x_gl]
[stopwords.append(x) for x in x_n]
[stopwords.append(x) for x in x_d]
[stopwords.append(x) for x in x_g]


def stemming_tokenizer(str_input):
    ##lower case, word tokenize,get rid of unnecessary symbols
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    ##merge hyphenated word by the new line
    words= [re.sub(r'-\n','', word) for word in words]
    ##spell check
    words = [spell.correction(word) for word in words]
    ###get rid of stopword
    words = [word for word in words if word not in stopwords]
    ##stemming
    words = [porter_stemmer.stem(word) for word in words]
    return words



 
#identify the vectorizer
vect = TfidfVectorizer(min_df=20, max_df=0.2,tokenizer=stemming_tokenizer)
#fit and transform documents to tfidf vectorizer
tfidf_vectorizer_vectors=vect.fit_transform(docs)

#get the first document
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame
tfidf= pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vect.get_feature_names(), columns=["tfidf"])
tfidf.sort_values(by=["tfidf"],ascending=False)

tfidf.to_excel('tfidf.xlsx')

###use price to predict which word has predicting power
##generate random price, 1 indicates price increase, 0 indicates price decrease

df['price'] = np.random.choice([0,1], df.shape[0])

###split train and test
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['txt'], 
                                                    df['price'], 
                                                    random_state=0)
###fit X_train with tfidf vectorizer
vect = TfidfVectorizer(tokenizer=stemming_tokenizer,stop_words=stopwords,min_df=5).fit(X_train)
len(vect.get_feature_names())


###use logistic regression model to train 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

###find out the words that contributes to price increase or decrease
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


