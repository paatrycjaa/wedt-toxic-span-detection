import pandas as pd
class DataProcessing:
    def load_data(self):
        return
    def extract_toxity(self):
        return
    def get_classes_amount(self, train_df):
        seriesObj = train_df.apply(lambda x: True if x.toxicity==1.0 else False, axis=1)
        numOfRows1 = len(seriesObj[seriesObj == True].index)
        numOfRows0 = len(seriesObj[seriesObj == False].index)
        return (numOfRows1, numOfRows0)
    def get_missing_class_elements(self,df, N, classValue):
        mask = df['toxicity'] == classValue
        new_df = df[mask]
        return new_df.sample(N)
