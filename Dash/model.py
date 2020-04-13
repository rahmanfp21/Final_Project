#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, matthews_corrcoef, auc, log_loss
from sklearn.model_selection import GridSearchCV
import pickle
from APPO import *

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
eng_stopwords = set(stopwords.words("english"))
pd.set_option('display.max_columns',None)
#

# Create a class to gain an information from text
def inside_info(df):
    # Percentage of capital letter to all letter
    df['exclamation'] = df['comment_text'].apply(lambda x: len(re.findall('!',x))/len(x))
    # Percentage of ! to all letter
    df['capital_letter_percent'] = df['comment_text'].apply(lambda x: len(re.findall('[A-Z]',x))/len(x))
    # Percentage of capital word to all word
    df['capital_word_percent'] = df['comment_text'].apply(lambda x: len([w for w in x.split() if w.isupper()])/len(x.split()))
    return df

# Create a class for cleaning text
def clean(comment):

    comment=comment.lower()
    comment = re.sub(r'\\n','',comment)
    comment = re.sub(r'wikipedia:.*#[\.\S]+[a-z]', '', comment)
    comment = re.sub(r'http[\.\S]+[\.\S]', '', comment)
    comment = re.sub(r'[0-9\.\:]*[0-9]', '', comment)
    comment = re.sub(r'file:.*jpg', '', comment)
    comment = re.sub(r'\.gif', '', comment)
    comment = re.sub(r'\[\[user(.*)\|', '', comment)
    comment = re.sub(r'www[\.\S]+[\.\S]','', comment)
    comment = re.sub(r'http', '', comment)
    
    # add wikipedia to reduce size
    comment = re.sub(r'wikipedia', '', comment)
    
    #Split the sentences into words
    words = tokenizer.tokenize(comment)
    
    whitelist = ["not", "no"]
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [word.split() for word in words]
    words = [item for sublist in words for item in sublist]
    words = [char for char in words if char not in string.punctuation]
    words = [word for word in words if (word not in eng_stopwords or word in whitelist) and len(word) > 1]
    words = [lem.lemmatize(word, "v") for word in words]
    
    clean_sent=" ".join(words)
    
    b = re.findall("[\'\sa-zA-Z]", clean_sent)
    clean_sent = ''.join(b)
    return(clean_sent)

def try_model(text):
    df_trial = pd.DataFrame(data=text, columns = ['comment_text'], index=[0])
    
    inside_info(df_trial)
    df_trial['clean_text'] = df_trial['comment_text'].apply(clean)
    df_trial = df_trial[['comment_text', 'exclamation', 'capital_word_percent', 'capital_letter_percent', 'clean_text']]
    pipe_vm = pickle.load(open('hate_model.sav', 'rb'))
	
    Predict = pipe_vm.predict(df_trial)
    # print(Predict)
    PredictProb = pipe_vm.predict_proba(df_trial)
    # print(PredictProb)
    
    if PredictProb[0][0] > 0.5:
        x = 'That comment is not a hate speech with probability of {} %'.format(PredictProb[0][0]*100)
    else:
        x = 'That comment is a hate speech with probability of {} %'.format(PredictProb[0][1]*100)
    return x