# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:21:59 2018
This Language detector model detects seven languages that largely use the Latin
script(English, German, French, Italian, Spanish, Portuguese, Dutch). It is 
implemented using a Multiclass Random Forest text classifier using words, 
bigrams, and trigrams as features. To optimize training and testing performance
by reducing number of features,a corpus of 300,000 sentences for each language 
is leveraged from Leipzig Corpus and the 50 most frequent words, bigrams, and 
trigrams are shortlisted as features. The dataframe creation is slightly 
complicated, but it is highly vectorized to speed up performance. All train and
test datapoints are then represented in the reduced feature-space. A model 
trained on 5,000 sentences from each language takes less than 2 minutes to 
train, and performs at 98% accuracy. 
To replicate the environment, place the following data files sourced from 
http://wortschatz.uni-leipzig.de/en/download in a directory, and assign that to 
'dirname'
1. deu_mixed-typical_2011_300K-sentences.txt
2. eng_news_2005_300K-sentences.txt
3. fra_mixed_2009_300K-sentences.txt
4. ita_mixed-typical_2017_300K-sentences.txt
5. nld_mixed_2012_300K-sentences.txt
6. por_newscrawl_2011_300K-sentences.txt
7. spa_news_2006_300K-sentences.txt

Novel ideas - shortlisting features based on frequency to speed up random forest
performance

Scope for improvement -  Need to prune feature space to further remove  
redundancies. One approach could be through the use of maximal substrings.
For eg - the trigram ' a ' will be a substring of ' a' always and can be removed

@author: Kiran Ramnath
Applicant ID - 201806110737_RamnathKiran
"""
import pandas as pd
from string import punctuation
import time
import numpy as np
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
#%%
dirname='C:/Personal/R/UKP/Data/'
#%% Read tab separated files

def read_file(path):

    t=pd.read_fwf(path, delimiter="\t",header=None)
    t[0]=t[0].apply(lambda row: row.split("\t")[1])

    return t
#%% function that takes text and n-gram length as input, returns list of tuples 
#   of the format [(ngram_count, n-gram)]    
    
def max_ngram_extracter(sent,num_of_chars):
    
    ngram_vectorizer=CountVectorizer(input="content",analyzer="char_wb",ngram_range=(num_of_chars,num_of_chars))
    ngrams=ngram_vectorizer.fit_transform(sent)
    
    count_values=ngrams.toarray().sum(axis=0)
    vocab = ngram_vectorizer.vocabulary_
    counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
    counts = [(c[1],c[0]) for c in counts]
    
    return counts
#%% function that takes text as input, returns list of tuples of the format 
#   [(word_count, word)]
    
def max_word_extracter(sent):

    word_vectorizer=CountVectorizer(input="content",analyzer="word", ngram_range=(1,1))
    ngrams=word_vectorizer.fit_transform(sent)
    
    count_values=ngrams.toarray().sum(axis=0)
    vocab = word_vectorizer.vocabulary_
    counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
    counts = [(c[1],c[0]) for c in counts]
    
    return counts    
#%% function that reads all csv files (one per language). Contains language
#   column that will be used as training class for the classification algorithm.
#   returns train and test dataframes, all tuples of bigrams, trigrams, count_words    

def language_df_creator(filename,language):    

    df=read_file(dirname+filename)
    df['language']=language
    
    #creating train-test split
    df_train, df_test = train_test_split(df, test_size=1000, random_state=42)
    
    #merging all text from train_df in one list to find most frequent words,bigrams and trigrams
    
    lang_txt=df_train.groupby('language')[0].apply(lambda row: (" ").join(row))
    lang_count_bigrams=max_ngram_extracter(lang_txt,2)
    lang_count_trigrams=max_ngram_extracter(lang_txt,3)
    lang_count_words=max_word_extracter(lang_txt)
    
    return df_train, df_test, lang_count_bigrams, lang_count_trigrams, lang_count_words
#%%
english_df_train, english_df_test, english_count_bigrams, english_count_trigrams, english_count_words = language_df_creator("eng_news_2005_300K-sentences.txt","english")

#%%
german_df_train, german_df_test, german_count_bigrams, german_count_trigrams, german_count_words=language_df_creator("deu_mixed-typical_2011_300K-sentences.txt","german")

#%%
italian_df_train, italian_df_test, italian_count_bigrams, italian_count_trigrams, italian_count_words=language_df_creator("ita_mixed-typical_2017_300K-sentences.txt","italian")

#%%
spanish_df_train, spanish_df_test, spanish_count_bigrams, spanish_count_trigrams, spanish_count_words=language_df_creator("spa_news_2006_300K-sentences.txt","spanish")

#%%
portuguese_df_train, portuguese_df_test, portuguese_count_bigrams, portuguese_count_trigrams, portuguese_count_words=language_df_creator("por_newscrawl_2011_300K-sentences.txt","portuguese")

#%%
french_df_train, french_df_test, french_count_bigrams, french_count_trigrams, french_count_words = language_df_creator("fra_mixed_2009_300K-sentences.txt","french")

#%%
dutch_df_train, dutch_df_test, dutch_count_bigrams, dutch_count_trigrams, dutch_count_words=language_df_creator("nld_mixed_2012_300K-sentences.txt","dutch")
#%% creating feature-list containing 50 most frequent words, bigrams, trigrams for all languages

features=set([f[0] for f in english_count_bigrams[0:50]
                        +english_count_words[0:50]
                        +english_count_trigrams[0:50]
                        +german_count_bigrams[0:50]
                        +german_count_trigrams[0:50]
                        +german_count_words[0:50]
                        +italian_count_bigrams[0:50]
                        +italian_count_trigrams[0:50]
                        +italian_count_words[0:50]
                        +spanish_count_bigrams[0:50]
                        +spanish_count_trigrams[0:50]
                        +spanish_count_words[0:50]
                        +portuguese_count_bigrams[0:50]
                        +portuguese_count_trigrams[0:50]
                        +portuguese_count_words[0:50]
                        +french_count_bigrams[0:50]
                        +french_count_trigrams[0:50]
                        +french_count_words[0:50]
                        +dutch_count_bigrams[0:50]
                        +dutch_count_trigrams[0:50]
                        +dutch_count_words[0:50]
                        if f[0] not in punctuation])

#%% Using Gensim's dictionary object to store features. This allows us to 
    #create dense vectors for all datapoints efficiently, speeding up data creation
    
dct=Dictionary([list(features)])

#re-assigning features because gensim creates dictionary in alphabetical order
features=list(dct.token2id.keys())
#%% 5,000 sentences from each each language is used to train the classification model 

english_df_train_frac=english_df_train.sample(n=5000,random_state=42)
german_df_train_frac=german_df_train.sample(n=5000,random_state=42)
italian_df_train_frac=italian_df_train.sample(n=5000,random_state=42)
spanish_df_train_frac=spanish_df_train.sample(n=5000,random_state=42)
portuguese_df_train_frac=portuguese_df_train.sample(n=5000,random_state=42)
french_df_train_frac=french_df_train.sample(n=5000,random_state=42)
dutch_df_train_frac=dutch_df_train.sample(n=5000,random_state=42)
#%%
train_df=pd.concat([english_df_train_frac,german_df_train_frac,
                    italian_df_train_frac,spanish_df_train_frac, 
                    portuguese_df_train_frac,french_df_train_frac,dutch_df_train_frac], 
                   ignore_index=True)
#%% Free up memory, perform garbage collection
del german_df_train, french_df_train, english_df_train, dutch_df_train, spanish_df_train, italian_df_train
gc.collect()
#%% create dataframe for all training sentences containing unique features as columns 

def create_dataframe_rf(df):

    df.rename(columns={0:"text"}, inplace=True)
    zero_data=np.zeros(shape=(len(df),len(features)))
    feature_df=pd.DataFrame(zero_data,index=df.index, columns=features)
    df=pd.concat([df,feature_df],axis=1)

    return df

train_df=create_dataframe_rf(train_df)
#train_df=pd.write_csv(dirname+"rftrain_set.csv")
#%% label encoder applies integer labels to all classes

languages=['english','dutch','german','italian','spanish','portuguese','french']

#create flags on the basis of language
le = preprocessing.LabelEncoder()    
le.fit(languages)     

#%% create features for all sentences, find and populate shortlisted feature columns
#   This is the most time consuming step. 

def feature_creator(df):    

    #Using Gensim utility function to populate training dataset
    bag_of_words=df['text'].apply(lambda row: sparse2full([(dct.token2id[m[0]],m[1]) for m in max_word_extracter([row]) if m[0] in dct.token2id], length=len(features)))
    bag_of_words=np.array(bag_of_words.tolist()).astype(int)

    bag_of_bigrams=df['text'].apply(lambda row: sparse2full([(dct.token2id[m[0]],m[1]) for m in max_ngram_extracter([row],2) if m[0] in dct.token2id], length=len(features)))        
    bag_of_bigrams=np.array(bag_of_bigrams.tolist()).astype(int)

    bag_of_trigrams=df['text'].apply(lambda row: sparse2full([(dct.token2id[m[0]],m[1]) for m in max_ngram_extracter([row],3) if m[0] in dct.token2id], length=len(features)))
    bag_of_trigrams=np.array(bag_of_trigrams.tolist()).astype(int)

    #Add matrix representation of words, bigrams, and trigrams components of all vectors
    all_features=bag_of_words+bag_of_bigrams+bag_of_trigrams
    all_features.astype(int)

    #populate train_df with all features
    all_features_df=pd.DataFrame(data=all_features, columns=features)
    df.update(all_features_df)
    
    #random forest requires integer labels, so transform text levels to integer levels
    df['flag']=le.transform(df['language'])           
    print ("Feature creation finished")
    
    return df
#%%
time_start=time.time()
train_df = feature_creator(train_df) 
time_taken_train_df=time.time()-time_start
#%%
def random_forest(train_df, number_of_estimators):
        
    # Random Forest Model 
    clf_rforest = RandomForestClassifier(n_estimators=number_of_estimators, random_state=1, min_samples_leaf=5, max_depth=30)
    clf_rforest.fit(train_df[features], train_df['flag'])

    # Importance of features
    rforest_importances = clf_rforest.feature_importances_
    rforest_importances = dict(zip(features,rforest_importances))

    return clf_rforest, rforest_importances

time_start=time.time()
clf_rforest, rforest_importances=random_forest(train_df, 500)
time_taken_rf=time.time()-time_start
#%% create test set containing 1000 sentences from each language

test_df=pd.concat([english_df_test,german_df_test,italian_df_test,spanish_df_test,
                        portuguese_df_test,french_df_test,dutch_df_test], ignore_index=True)
#%%
# Process of creating features for scoring dataset
def scoring_df(test_df):

    test_df=create_dataframe_rf(test_df)
    test_df=feature_creator(test_df)

    return test_df
#%%
time_start=time.time()
test_df=scoring_df(test_df)
time_taken_test_df=time.time()-time_start
#%%
time_start=time.time()
test_df['prediction']=le.inverse_transform(clf_rforest.predict(test_df[features]))
time_taken_score=time.time()-time_start    
#%%
# Exporting Scored Results to CSV
test_df.to_csv(dirname+"preds_5k.csv")
#%% Print performance metrics like precision, recall, f-1 score

print(classification_report(test_df['language'],test_df['prediction'],target_names=languages))
#%% Print confusion matrix

conf_mat = confusion_matrix(test_df['language'],test_df['prediction'])
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=languages, yticklabels=languages)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()