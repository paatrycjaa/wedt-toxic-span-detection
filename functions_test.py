
from src.SemEvalData import SemEvalData
from src.preprocessing import  getSpansByToxicWords

MAX_WORD_NUM = 100 

train_data_semeval = SemEvalData(MAX_WORD_NUM)
data = train_data_semeval.load_data("data/tsd_trial.csv")
train_df_preprocessed = train_data_semeval.preprocess()
withChar = train_df_preprocessed.iloc[0]
withoutChar = train_df_preprocessed.iloc[1]

span1 = getSpansByToxicWords(withChar['toxic_words'], withChar['text'])
print('span1',withChar, span1)
span2 = getSpansByToxicWords(withoutChar['toxic_words'], withoutChar['text'], withoutChar['diff'])

print(withoutChar, span2)