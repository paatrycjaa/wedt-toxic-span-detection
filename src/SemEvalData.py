import pandas as pd
from ast import literal_eval
import numpy as np
from src.WordsExtraction import WordsExtraction
from src.preprocessing import clean_str, preprocess_bayes
from src.DataProcessig import DataProcessing
from nltk import tokenize

class SemEvalData(DataProcessing):
    def __init__(self, MAX_WORD_NUM=40):
        self.MAX_WORD_NUM = MAX_WORD_NUM

    def load_data(self, path):
        self.train = pd.read_csv(path) ##"data/tsd_trial.csv"
        self.train["spans"] = self.train.spans.apply(literal_eval)
        return self.train
    
    def preprocess(self):
        self.train["toxicity"] = ( self.train.spans.map(len) > 0 ).astype(int)
        words_extractor = WordsExtraction()
        self.train['toxic_words'] = self.train.apply(lambda row: words_extractor.extractToxicWordIndexUsingSpans(row), axis=1)
        ## clean text
        self.train['text'] = self.train.apply(lambda row: preprocess_bayes(row.text), axis=1)
        ## clean toxic words - previously it was only clean_str
        self.train['toxic_words'] = self.train.apply(lambda row: [preprocess_bayes(word) for word in row.toxic_words], axis =1 )
        ## extract senteces
        self.train['sentences'] = self.train.apply(lambda row: tokenize.sent_tokenize(row.text), axis=1)

        ## toxity per sentence
        self.train['toxicity_sentence'] = self.train.apply(lambda row: self.__extract_toxity(row.toxic_words, row.sentences), axis = 1)
        
        return self.train

    def __extract_toxity(self,toxic_words, sentences):
        toxicity = []
        for sentence in sentences:
            if any(word in sentence for word in toxic_words):
                # toxicity.append(np.full( self.MAX_WORD_NUM, 1.0, dtype='float32'))
                toxicity.append(1.0)
            else:
                # toxicity.append(np.full(self.MAX_WORD_NUM, 0.0, dtype='float32'))
                toxicity.append(0.0)
        return toxicity