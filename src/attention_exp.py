"""
Project Toxic Span Detection
Postprocessing for model LSTM with attention
@authors: Julia Kłos, Patrycja Cieplicka
@date: 12.01.2020
"""
import numpy as np

def getWordsByAttention(attention, tokenized, word_vect, treshold):
    """
    Function returning toxic words, given word vector, attention weights, word_indexes and treshold
    """
    tokenized = tokenized[0] > 0
    weights = attention[tokenized]
    weights = weights > treshold
    words = [word for k, word in enumerate(word_vect.split(' '))]
    toxic_words = [b for a, b in zip(weights, words) if a]
    return toxic_words

def wordAttentionWeights(sequenceSentence,weights):
    """
    The same function as the AttentionLayer class - calculate the weights of attention layer
    """
    uit = np.dot(sequenceSentence, weights[0]) + weights[1]
    uit = np.tanh(uit)

    ait = np.dot(uit, weights[2])
    ait = np.squeeze(ait)
    ait = np.exp(ait)
    ait /= np.sum(ait)
    
    return ait