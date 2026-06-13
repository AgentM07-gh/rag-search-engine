import string

import os

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, load_movies, CACHE_DIR

import pickle


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results



def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text):
        token_list = tokenize_text(text)
        for token in token_list:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}
    
    def get_documents(self, term)->list:
        doc_ids = list(self.index.get(term, set()))
        sorted_doc_ids = sorted(doc_ids)
        return sorted_doc_ids
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie
    
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        index_file = CACHE_DIR + "/index.pkl"
        docmap_file = CACHE_DIR + "/docmap.pkl"
        with open(index_file, "wb") as idx_file:
            pickle.dump(self.index, idx_file)
        with open(docmap_file, "wb") as dm_file:
            pickle.dump(self.docmap, dm_file)

def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    doc = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {doc[0]}")