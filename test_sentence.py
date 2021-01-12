"""
Project Toxic Span Detection
Implementation of functions returning toxic spans fot given text
@authors: Julia Kłos, Patrycja Cieplicka
@date: 12.01.2020
"""

import joblib
from src.preprocessing import preprocess_bayes, getSpansByToxicWords, getToxicWordsBayes, clean_str, get_diff
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import keras
from keras.models import load_model, Model
import numpy as np
import pickle
from src.attention_exp import getWordsByAttention, wordAttentionWeights
import pickle
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


# Declaring important variables
MAX_WORD_NUM = 40
MAX_FEATURES = 200000
TRESHOLD = 0.015
MAX_WORD_NUM_2 = 40
TRESHOLD_LSTM = -0.05

# Loading models
# Bayes
count_vect = joblib.load('vectorizer_bayes.jbl')
transformer = joblib.load('transformer_bayes.jbl')
model_bayes = joblib.load('model_bayes.jbl')
lemmer = nltk.stem.WordNetLemmatizer()

# LSTM-Attention
model_attention = keras.models.load_model('attention_model')
hidden_word_encoding_out = Model(inputs=model_attention.input, outputs= model_attention.get_layer('dense').output)
word_context = model_attention.get_layer('attention').get_weights()

with open('tokenizer_nn.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# LSTM-Lime
model_lime = keras.models.load_model("lstm_pooling")
c = make_pipeline(Transform(tokenizer),model_lime)

class Transform():
    """
    Class preprocessing and converting input data into vectors for LIME explanation
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print("")
    
    def fit(self, X, y = None):
        print("")
        return self
    
    def transform(self, X, y =None):
        X_ = vectorize(X, self.tokenizer)
        return X_



def getPredictedWordsFromSentence(sentence, threshold,c):
    """
    Function extracting toxic words for LIME model explanation
    """
    explainer = LimeTextExplainer(class_names=["NoToxic","Toxic"])
    exp = explainer.explain_instance(sentence,c.predict_proba, num_features=6, top_labels = 2)
    expWords = exp.as_list()
    maxScore = max(expWords, key = lambda i : i[1])[0]
    expWords = filter(lambda t: t[1] < threshold, expWords )
    wordsList = [i[0] for i in expWords]
    wordsList.append(maxScore) if maxScore not in wordsList else wordsList
    return wordsList

def test_sentence(text, classifierType):
    """
    Choosing type of classifier to extract toxic spans
    """
    span = []
    if(classifierType =="bayes"):
        span = test_bayes(text)
    elif (classifierType == "lime"):
        span = test_lime(text)
    elif (classifierType == "attention"):
        span = test_attention(text)
    return span
    
def test_bayes(text, treshold):
    """
    Function returning toxic span using Bayes classifier
    """
    tokenized = tokenize.sent_tokenize(text)
    sentences = [preprocess_bayes(sentence) for sentence in tokenized]
    toxic_words = []
    for sentence in sentences:
        data = ' '.join([lemmer.lemmatize(word) for word in sentence.split()])
        x = count_vect.transform([data])
        y = model_bayes.predict(x)
        if y == 1.0:
            toxic = getToxicWordsBayes(count_vect, x, treshold)
            toxic_words = [*toxic_words, *toxic]

    text_preprocessed = preprocess_bayes(text)
    diff = get_diff(text, text_preprocessed)        
    spans = getSpansByToxicWords(toxic_words,text_preprocessed, diff)
    return spans

def vectorize(x, tokenizer):
    """
    Function converting text data to number vectors using tokenizer
    """
    if not isinstance(x, list): 
        x = [x]
    data_temp = np.zeros((len(x), MAX_WORD_NUM_2), dtype='int32')
    for i, sentence in enumerate(x):
        for k, word in enumerate(sentence):
            try:
                if k<MAX_WORD_NUM and tokenizer.word_index[word]<MAX_FEATURES:
                    data_temp[i,k] = tokenizer.word_index[word]
            except:
                pass
    return data_temp

def preprocess_lstm(text):
    """ Function preprocessing data for LSTMs models"""
    text = clean_str(text)
    tokenized = tokenize.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in tokenized]
    
    for i, w in enumerate(sentences):
        sentences[i] = [word for word in sentences[i] if word.isalpha()]

    sentences = [x for x in sentences if x!=[]]
    return text, tokenized, sentences

def test_lime(text):
    """
       Function returning toxic span using LSTM classifier and LIME metod explanation
    """
    text_cleaned, tokenized, sentences = preprocess_lstm(text)
    vect = vectorize(sentences, tokenizer)
    y_pred = model_lime.predict(vect)
    toxic_words  = []
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == 0 :
            text_ = ' '.join(sentences[i])
            toxic = getPredictedWordsFromSentence(text_, TRESHOLD_LSTM,c)
            toxic_words = toxic_words + toxic
    diff = get_diff(text, text_cleaned)        
    spans = getSpansByToxicWords(toxic_words, text_cleaned,diff)
    print(spans)
    return spans

def test_attention(text):
    """
       Function returning toxic span using LSTM wih Attention classifier
    """
    text_cleaned, tokenized, sentences = preprocess_lstm(text)
    vect = vectorize(sentences, tokenizer)
    toxic_words = []
 
    
    for i in range(len(sentences)):
        in_data = vect[i].reshape(1,MAX_WORD_NUM)
        y = model_attention.predict(in_data)
        Y= np.where(y > 0.5,1,0)
        if Y == 1 :
            text_ = ' '.join(sentences[i])
            hidden_word_encodings = hidden_word_encoding_out.predict(in_data)
            # Compute context vector using output of dense layer
            ait = wordAttentionWeights(hidden_word_encodings,word_context)
            toxic = getWordsByAttention(ait,in_data, text_,TRESHOLD)
            toxic_words = toxic_words+toxic
    diff = get_diff(text, text_cleaned)        
    spans = getSpansByToxicWords(toxic_words, text_cleaned, diff)

    return spans

if __name__ == "__main__":
    nltk.download('wordnet')
    span = test_sentence("Another idiot!", "attention")
    span = test_sentence("To be or not be that is a question", "attention")
    print(span)
    #span = test_sentence("These freaking donkeys all need to be removed from office. I'm so sick and tired of these lifelong politicians who all seem clueless and could never run their own business.", "attention")