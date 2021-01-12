"""
Project Toxic Span Detection
Implementation of functions for preprocessing and postprocessing
@authors: Julia Kłos, Patrycja Cieplicka
@date: 12.01.2020
"""

import re
import numpy as np
import os
import string
import difflib


def preprocess_bayes(text):
    """
    Function preprocessing data for Bayes model
    """
    text = clean_str(text)
    # Remove punctation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove digits
    text = re.sub('\d', '', text)
    return text

def clean_str(text):
    """
    Function string cleaning for dataset
    Every dataset is lower cased except
    """
    text = re.sub(r"\\", "", text)   
    text = re.sub(r"\'", "", text)  
    text = re.sub(r"\"", "", text)
    text = re.sub("\\n"," ", text) 
    return text.strip().lower()

def getSpansByToxicWords(toxicwords, sentence, positions=[]):
    """
    Function returning spans given words
    """
    spans = []
    for word in toxicwords:
        if(len(word)> 1):
            start = sentence.find(word)
            end = start + len(word)
            for pos in positions:
                if(pos <= end):
                    end = end+1
                if(pos < start):
                    start = start + 1     
            span = [*range(start, end, 1)]
            spans = spans + span
    return spans

def getToxicWordsBayes(vectorizer,vect, treshold):
    """
    Function returning toxic words for Bayes given features and treshold
    """
    words = vectorizer.get_feature_names()
    array = vect.todense().getA()
    i = 0;
    num = []
    for val in array[0]:
        if val > treshold :
            num.append(i)
        i=i+1
    
    return [words[j] for j in num]
def get_diff(original, cleaned):
    """
    Function helping getting spans
    """
    original =  re.sub("\\n"," ", original) 
    lower_cased = original.lower()
    positions = []
    for i,s in enumerate(difflib.ndiff(lower_cased, cleaned)):
        if s[0]==' ': continue
        else: positions.append(i)
    return positions
    