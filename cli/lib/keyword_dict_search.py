import json
import os
import string


def keyword_movie_search(search: str)->list:

    file_path = os.path.abspath("data/movies.json")
    
    with open(file_path, "r") as file:
        movie_dict = json.load(file)

    search_results = []

    counter = 0
    for key, movie_list in movie_dict.items():
 #       print(key)
        for movie in movie_list:
            punc_rem = str.maketrans("","",string.punctuation)
            srch_str = search.lower().translate(punc_rem)
            title_str = movie.get("title").lower().translate(punc_rem)
            if srch_str in title_str and counter < 5:
                counter += 1
                search_results.append(movie.get("title"))

    return search_results