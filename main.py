import pandas as pd
from src.WordsExtraction import WordsExtraction
from ast import literal_eval

## load data
train = pd.read_csv("data/tsd_trial.csv")
train["spans"] = train.spans.apply(literal_eval)


## set category 1- toxic, 0- non-toxic
train["toxic"] = ( train.spans.map(len) > 0 ).astype(int)
train = train.iloc[0:6]
## extract toxic words
words_extractor = WordsExtraction()
train['toxic_words'] = train.apply(lambda row: words_extractor.extractToxicWordIndexUsingSpans(row), axis=1)
first = train.iloc[1]

print(train.head(7), first)
print(train.spans.iloc[0])