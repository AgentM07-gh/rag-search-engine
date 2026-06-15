import string
import sys
import os

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, load_movies, CACHE_DIR

import pickle


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

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
    
    def load(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        index_file = CACHE_DIR + "/index.pkl"
        docmap_file = CACHE_DIR + "/docmap.pkl"
        try:
            with open(index_file, "rb") as idx_file:
                self.index = pickle.load(idx_file)
        except FileNotFoundError:
            print(f"Error: The file '{index_file}' was not found.")
            sys.exit(1)
        try:
            with open(docmap_file, "rb") as dm_file:
                self.docmap = pickle.load(dm_file)       
        except FileNotFoundError:
            print(f"Error: The file '{index_file}' was not found.")
            sys.exit(1) 

def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()