import json
import os
import string
from lib.tokenization import token_split
from lib.stopwords_removal import remove_stopwords
from nltk.stem import PorterStemmer


def keyword_movie_search(search: str)->list:

    file_path = os.path.abspath("data/movies.json")
    
    with open(file_path, "r") as file:
        movie_dict = json.load(file)

    search_results = []

    stemmer = PorterStemmer()

    counter = 0
    for key, movie_list in movie_dict.items():
 #       print(key)
        for movie in movie_list:
            srch_list = remove_stopwords(token_split(search))
            title_list = remove_stopwords(token_split(movie.get("title")))

            search_stems = []
            for search in srch_list:
                search_stems.append(stemmer.stem(search))
            
            title_stems = []
            for title in title_list:
                title_stems.append(stemmer.stem(title))
           
            if any(srch in title for srch in search_stems for title in title_stems) and counter < 5:
                counter += 1
                search_results.append(movie.get("title"))

    return search_results