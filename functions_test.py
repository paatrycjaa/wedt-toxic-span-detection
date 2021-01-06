
from src.SemEvalData import SemEvalData

MAX_WORD_NUM = 100 

train_data_semeval = SemEvalData(MAX_WORD_NUM)
data = train_data_semeval.load_data("data/tsd_trial.csv")
train_df_preprocessed = train_data_semeval.preprocess()

print(train_df_preprocessed.head(20))