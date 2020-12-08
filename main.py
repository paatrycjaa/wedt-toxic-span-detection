import pandas as pd
from src.WordsExtraction import WordsExtraction
from ast import literal_eval
from nltk import tokenize
import nltk
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
import re


nltk.download('punkt')
## load data
train = pd.read_csv("data/tsd_trial.csv")
train["spans"] = train.spans.apply(literal_eval)


## set category 1- toxic, 0- non-toxic
train["toxicity"] = ( train.spans.map(len) > 0 ).astype(int)
train = train.iloc[0:6]
## extract toxic words
words_extractor = WordsExtraction()
train['toxic_words'] = train.apply(lambda row: words_extractor.extractToxicWordIndexUsingSpans(row), axis=1)

def clean_str(string):
    """
    string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\\n","", string)    
    return string.strip().lower()

def extract_toxity(toxic_words, sentences):
    toxicity = []
    for sentence in sentences:
        if any(word in sentence for word in toxic_words):
            toxicity.append(1)
        else:
            toxicity.append(0)
    return toxicity

paras = []
labels = []
texts = []
sent_lens = []
sent_nums = []
## clean text
train['text'] = train.apply(lambda row: clean_str(row.text), axis=1)
## clean toxic words
train['toxic_words'] = train.apply(lambda row: [clean_str(word) for word in row.toxic_words], axis =1 )
## extract senteces
train['sentences'] = train.apply(lambda row: tokenize.sent_tokenize(row.text), axis=1)

## toxity per sentence
train['toxicity_sentence'] = train.apply(lambda row: extract_toxity(row.toxic_words, row.sentences), axis = 1)

first = train.iloc[1]
train_data = {'sentence':  train.sentences.sum(),
        'toxicity_sentence': train.toxicity_sentence.sum()
        }

train_df = pd.DataFrame (train_data, columns = ['sentence','toxicity_sentence'])
print(train.head(7), first.sentences[0])
print(train.spans.iloc[0], train_df)
