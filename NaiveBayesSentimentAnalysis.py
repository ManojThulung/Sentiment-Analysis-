#!/usr/bin/env python
# coding: utf-8

# In[156]:


import numpy as np
import pandas as pd

import re  #library for regular expression operation
import string  #for string operation

from nltk.corpus import stopwords #module for stop words that come with NLTK
from nltk.stem import PorterStemmer #module for stemming
from nltk.tokenize import TweetTokenizer #module for tokenizing strings


file = open("Reddit_Data.csv", "r", encoding="utf8")
data = file.read().split("\n")
data = [d.split(",") for d in data]
del(data[0]) #remove the column title from


data = pd.DataFrame(data)
data.columns = ["text","label"]
data.head(7)


#Divided the set of positive, neutral and negative data
positive_data = []
neutral_data = []
negative_data = []

for each in data.index:
    if data["label"][each] == "1":
        positive_data.append(data["text"][each].strip())
    elif data["label"][each] == "0":
        neutral_data.append(data["text"][each].strip())
    elif data["label"][each] == "-1":
        negative_data.append(data["text"][each].strip())

print('Numer of Positive data: ',len(positive_data))
print('Numer of Neutral data: ',len(neutral_data))
print('Numer of Negative data: ',len(negative_data))


# ## PreProcess Data
# ### Remove hyperlinks, styles


def remove_hyperlinks_marks_styles(data):
    #remove old style retweet text "RT"
    new_data = re.sub(r'^RT[\s]+', '', data)
    
    #remove hyperlinks
    new_data = re.sub(r'https?:\/\/.*[\r\n]*', '', new_data)
    
    #remove hastags
    #only removing the hast # sign from the word
    new_data = re.sub(r'#', '', new_data)
    
    return new_data


# Testing remove_hyperlinks_marks_styles(data) function
dummy_data = "@Mish23615351  follow @jnlazts &amp; http://t.co/RCvcYYO0Iq follow u back :)"
print(dummy_data)
print()
print(remove_hyperlinks_marks_styles(dummy_data))


# ### Tokenize the string
# Spliting a string of sentence into individual words


# Instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def tokenize_data(data):
    data_tokens = tokenizer.tokenize(data)
    return data_tokens

# testing function tokenize_data(data)
print(neutral_data[0])
print()
print(tokenize_data(neutral_data[0]))


# ### Removing stop words and punctuation
# Stop words are the words that do not add significant meaning to the text. For example "i" and "me"


import nltk
nltk.download('stopwords')

# Importing the english stop words list from NLTK
stopwords_english = stopwords.words('english')

punctuations = string.punctuation

def remove_stopwords_punctuations(data_tokens):
    data_clean = []
    
    for word in data_tokens:
        if (word not in stopwords_english and word not in punctuations):
            data_clean.append(word)
    
    return data_clean


# testing remove_stopwords_punctuations(data_tokens) function
dummy_token = tokenize_data(neutral_data[0])
print(dummy_token)
print()
print(remove_stopwords_punctuations(dummy_token))


# ### Stemming 
# It is the process of converting a word to the most general form for eg
# 
# passing = pass,
# passed = pass

stemmer = PorterStemmer()

def get_stem(data_clean):
    data_stem = []
    
    for word in data_clean:
        stem_word = stemmer.stem(word)
        data_stem.append(stem_word)
    
    return data_stem


# Testing get_stem(data_clean) function
dummy_clean = ['passing', 'passed', 'pass', 'doing', 'do']
print(dummy_clean)
print()
print(get_stem(dummy_clean))


# ### Combining all preprocess techniques
# 


def process_data(data):
    processed_data = remove_hyperlinks_marks_styles(data)
    data_tokens = tokenize_data(processed_data)
    data_clean = remove_stopwords_punctuations(data_tokens)
    data_stem = get_stem(data_clean)
    
    return data_stem


# testing process_data(data) function
print(neutral_data[0])
print()
print(process_data(neutral_data[0]))


# ### Splitting data into training and testing dataset
# 80% of data is divided to training dataset and 20% to testing dataset

test_pos = positive_data[11080:]
train_pos = positive_data[:11080]

test_neu = neutral_data[9210:]
train_neu = neutral_data[:9210]

test_neg = negative_data[5800:]
train_neg = negative_data[:5800]


train_x = train_pos + train_neu + train_neg
test_x = test_pos + test_neu + test_neg

# Corrosponding dataset values, positive as 1, neutral as 0 and negative as -1
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neu)))
train_y = np.append(train_y, -np.ones(len(train_neg)))

test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neu)))
test_y = np.append(test_y, -np.ones(len(test_neg)))


# ### Creating Frequency dictionary


def create_frequency(data, ys):
    freq_d = {}
    
    #Create frequency dictionary
    for d, y in zip(data, ys):
        for word in process_data(d):
            pair = (word, y)
            
            if pair in freq_d:
                freq_d[pair] += 1
            else:
                freq_d[pair] = freq_d.get(pair, 1)
    return freq_d


# testing create_frequency() function
data = ['this sad movie is sad and horror', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]

freq_d = create_frequency(data, ys)
print(freq_d)


# ### Training Model using Naive Bayes


# Building frequency dictionary
freqs = create_frequency(train_x, train_y) # Please have patience, this will take a minute

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    
    #calculating the number of unique words in freqs
    unique_words = set([pair[0] for pair in freqs.keys()])
    V = len(unique_words)
    
    #calculating N_pos, N_neu and N_neg (total number of words in training dataset)
    N_pos = N_neu = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[(pair)]
        elif pair[1] == 0:
            N_neu += freqs[(pair)]
        else:
            N_neg += freqs[(pair)]
            
    #calculating the number of data
#     D = len(train_y)
    D = train_y.shape[0] #gives number of row count

    #calculating the number of positive, neutral and negative data
    D_pos = D_neu = D_neg = 0
    for each in train_y:
        if each == 1:
            D_pos += each
        elif each == -1:
            D_neg += each
    #since the D_neg is in negative value, it is converted to positive
    D_neg = abs(D_neg)
    D_neu = D-(D_pos + D_neg)
    
    #calculating logprior
    logprior = np.log(D_pos) - np.log(D_neu)

    for word in unique_words:
        freq_pos = freqs.get((word, 1), 0)
        freq_neu = freqs.get((word, 0), 0)
        freq_neg = freqs.get((word, -1), 0)
        
        #calculating the probability that word in positive, neutral and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neu = (freq_neu + 1) / (N_neu + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        
        #Calculating the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
        
    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)


# ### Predicting data
# 


def naive_bayes_predict(data, logprior, loglikelihood):
    #Preprocessing the data to get list of words
    word_l = process_data(data)
    
    #initializaion probability to zero
    p = 0
    
    #adding the logprior
    p += logprior
    
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
            
    p = round(p, 0)
    if p > 0:
        return 1.0
    elif p == 0:
        return 0.0
    else:
        return -1.0


# this will take 10 to 15 sec.
result_data = []
for each in test_x:
    result_data.append(naive_bayes_predict(each, logprior, loglikelihood))


from sklearn.metrics import accuracy_score

score_naive = accuracy_score(test_y,result_data)
print()
print("Accuray of the model is", score_naive * 100,"%")




