"""
Project Toxic Span Detection
Implementation of class preprocessing input data from SemEval into dataframe
@authors: Julia Kłos, Patrycja Cieplicka
@date: 12.01.2020
"""

import pandas as pd
from ast import literal_eval
import numpy as np
from src.WordsExtraction import WordsExtraction
from src.preprocessing import clean_str, preprocess_bayes, get_diff
from nltk import tokenize

class SemEvalData(DataProcessing):
    """
    Class representing SemEvalData and preprocessing the data into dataframe
    """
    def __init__(self, MAX_WORD_NUM=40):
        self.MAX_WORD_NUM = MAX_WORD_NUM

    def load_data(self, path):
        self.train = pd.read_csv(path) ##"data/tsd_trial.csv"
        self.train["spans"] = self.train.spans.apply(literal_eval)
        return self.train
    
    def preprocess(self):
        null_check = self.train.isnull().sum()
        self.train["text"].fillna("unknown", inplace=True)
        self.train["toxicity"] = ( self.train.spans.map(len) > 0 ).astype(int)
        words_extractor = WordsExtraction()
        self.train['toxic_words'] = self.train.apply(lambda row: words_extractor.extractToxicWordIndexUsingSpans(row), axis=1)
        self.train['original_text'] = self.train['text']
        ## clean text
        self.train['text'] = self.train.apply(lambda row: clean_str(row.text), axis=1)
        ## clean toxic words - previously it was only clean_str
        self.train['toxic_words'] = self.train.apply(lambda row: [clean_str(word) for word in row.toxic_words], axis =1 )
        ## extract senteces
        self.train['sentences'] = self.train.apply(lambda row: tokenize.sent_tokenize(row.text), axis=1)
        self.train['diff'] = self.train.apply(lambda row: get_diff(row.original_text, row.text), axis= 1)

        ## toxity per sentence
        self.train['toxicity_sentence'] = self.train.apply(lambda row: self.__extract_toxity(row.toxic_words, row.sentences), axis = 1)
    
        return self.train

    def __extract_toxity(self,toxic_words, sentences):
        toxicity = []
        for sentence in sentences:
            if any(word in sentence for word in toxic_words):
                toxicity.append(1.0)
            else:
                toxicity.append(0.0)
        return toxicity
    def get_classes_amount(self, train_df):
        return super().get_classes_amount(train_df)
    def get_missing_class_elements(self,df, N, classValue):
        return super().get_missing_class_elements(df, N, classValue)