import os
import string

from lib.tokenization import token_split

def remove_stopwords(input_list: list)->list:

    file_path = os.path.abspath("data/stopwords.txt")
    
    with open(file_path, "r") as file:
        stopword_file = file.read()
    
    stopword_list = stopword_file.splitlines()
    punc_rem = str.maketrans("","",string.punctuation)
    
    for i in range(len(stopword_list)):
        stopword_list[i] = stopword_list[i].lower().translate(punc_rem)

   
    cleaned_list = [word for word in input_list if word not in set(stopword_list)]

    return cleaned_list