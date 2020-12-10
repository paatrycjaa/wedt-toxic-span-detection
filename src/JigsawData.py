import pandas as pd
from ast import literal_eval
from src.DataProcessig import DataProcessing
from src.preprocessing import clean_str
from nltk import tokenize
import numpy as np

class JigsawData(DataProcessing):
    def __init__(self, MAX_WORD_NUM=40):
        self.MAX_WORD_NUM = MAX_WORD_NUM

    def load_data(self, path):
        self.train = pd.read_csv(path) ##"data/tsd_trial.csv"
        return self.train

    def preprocess(self):
        null_check = self.train.isnull().sum()
        self.train["comment_text"].fillna("unknown", inplace=True)
        self.train = self.__clean_spam(self.train)
        self.train['toxicity'] = self.train.apply(lambda row: self.__extract_toxicity(row), axis=1)
        self.train['text'] = self.train.apply(lambda row: clean_str(row.comment_text), axis=1)
        ## extract senteces
        self.train['sentences'] = self.train.apply(lambda row: tokenize.sent_tokenize(row.text), axis=1)

        ## toxity per sentence
        self.train['toxicity_sentence'] = self.train.apply(lambda row: self.__extract_toxicity_per_sentence(row.sentences, row.toxicity), axis = 1)
        return self.train

    def __clean_spam(self,df):
        
        df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
        df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
        df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
        df=df[df['word_unique_percent']>30]
        return df
    def __extract_toxicity_per_sentence(self, sentence, toxicity):
        # return 
        return toxicity
    def __extract_toxicity(self, row):
        if(row['toxic']+row['severe_toxic']+row['obscene']+ row['threat']+ row['insult'] + row['identity_hate'] > 0):
            return 1
        else:
            return 0