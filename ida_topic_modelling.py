# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:37:16 2020

@author: zzhu1
"""

import pickle
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker   
import pandas as pd

# Load the list of documents
df=pd.read_excel('news_sample.xlsx')
df.dropna(inplace=True)
##check tfidf for each tokenized words
docs=df['txt']

##define tokenize, stemming and stopwords
spell = SpellChecker() 
porter_stemmer = PorterStemmer()
with open('StopWords_GenericLong.txt', 'r') as f:
    x_gl = f.readlines()
with open('StopWords_Names.txt', 'r') as f:
    x_n = f.readlines()
with open('StopWords_DatesandNumbers.txt', 'r') as f:
    x_d = f.readlines()
with open('StopWords_Geographic.txt', 'r') as f:
    x_g = f.readlines()
##extend stopwords like bi's paper

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
 
# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = TfidfVectorizer(min_df=20, max_df=0.2,tokenizer=stemming_tokenizer)
#fit and transform documents to tfidf vectorizer
X = vect.fit_transform(docs)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`
ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=10,id2word=id_map,passes=25,random_state=34)

def lda_topics():
    
    # Your Code Here
    lis=ldamodel.show_topics(num_topics=10,num_words=10)
    
    return lis 
lda_topics()












