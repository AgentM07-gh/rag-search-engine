import string
import sys
import os
import math

from nltk.stem import PorterStemmer
from collections import Counter, defaultdict

from .search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    STOPWORDS_PATH, 
    load_movies, 
    CACHE_DIR,
    BM25_K1,
    BM25_B
)

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

def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    token = single_token(term)
    return idx.get_tf(doc_id, token)

def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = single_token(term)
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.index[token])
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = single_token(term)
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.index[token])
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return inverted_index.term_frequencies[doc_id][token] * idf

def bm25_idf_command(term: str)->float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = single_token(term)
    bm25_idf = inverted_index.get_bm25_idf(token)
    return bm25_idf

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()    
    token = single_token(term)
    bm25_tf = inverted_index.get_bm25_tf(doc_id, token, k1, b)
    return bm25_tf

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

def single_token(text)->str:
    token = tokenize_text(text)
    if len(token) != 1:
        raise ValueError("Must return only one token")
    return token[0]

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.doc_lengths = {}
        self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.termfreq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        token_list = tokenize_text(text)
        self.doc_lengths[doc_id] = len(token_list)
        for token in token_list:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] +=1
    
    def get_documents(self, term)->list:
        doc_ids = list(self.index.get(term, set()))
        sorted_doc_ids = sorted(doc_ids)
        return sorted_doc_ids
    
    def get_tf(self, doc_id, term)->int:
        return self.term_frequencies[doc_id][term]

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie
    
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as idx_file:
            pickle.dump(self.index, idx_file)
        with open(self.docmap_path, "wb") as dm_file:
            pickle.dump(self.docmap, dm_file)
        with open(self.termfreq_path, "wb") as tf_file:
            pickle.dump(self.term_frequencies, tf_file)
        with open(self.doc_lengths_path, "wb") as dl_file:
            pickle.dump(self.doc_lengths, dl_file)

    def load(self):
        try:
            with open(self.index_path, "rb") as idx_file:
                self.index = pickle.load(idx_file)
        except FileNotFoundError:
            print(f"Error: The file '{self.index_path}' was not found.")
            sys.exit(1)
        try:
            with open(self.docmap_path, "rb") as dm_file:
                self.docmap = pickle.load(dm_file)       
        except FileNotFoundError:
            print(f"Error: The file '{self.docmap_path}' was not found.")
            sys.exit(1)
        try:
            with open(self.termfreq_path, "rb") as tf_file:
                self.term_frequencies = pickle.load(tf_file)       
        except FileNotFoundError:
            print(f"Error: The file '{self.termfreq_path}' was not found.")
            sys.exit(1) 
        try:
            with open(self.doc_lengths_path, "rb") as dl_file:
                self.doc_lengths = pickle.load(dl_file)       
        except FileNotFoundError:
            print(f"Error: The file '{self.doc_lengths_path}' was not found.")
            sys.exit(1) 

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index[term])
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) +1)
        return bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
    
    def __get_avg_doc_length(self) -> float:
        avg_doc_len = 0.0
        if len(self.doc_lengths) > 0:
            avg_doc_len = sum(v for v in self.doc_lengths.values())/len(self.doc_lengths)
        return avg_doc_len


