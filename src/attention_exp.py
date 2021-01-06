import numpy as np
### na etapie wyciagania spans- dodac, ze jesli klasa 0 to pusty span
def getWordsByAttention(attention, tokenized, word_vect, treshold):
    tokenized = tokenized[0] > 0
    weights = attention[tokenized]
    weights = weights > treshold
    words = [word for k, word in enumerate(word_vect.split(' '))]
    toxic_words = [b for a, b in zip(weights, words) if a]
    return toxic_words

def wordAttentionWeights(sequenceSentence,weights):
    """
    The same function as the AttentionLayer class.
    """
    uit = np.dot(sequenceSentence, weights[0]) + weights[1]
    uit = np.tanh(uit)

    ait = np.dot(uit, weights[2])
    ait = np.squeeze(ait)
    ait = np.exp(ait)
    ait /= np.sum(ait)
    
    return ait