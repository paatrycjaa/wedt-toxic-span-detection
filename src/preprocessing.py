import re
def clean_str(string):
    """
    string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    string = re.sub("\\n"," ", string)    
    return string.strip().lower()
