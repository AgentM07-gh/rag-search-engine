#!/usr/bin/env python3

import argparse
import sys
from lib.keyword_search import (
    search_command, 
    build_command, 
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command
)
from lib.search_utils import (
    BM25_K1,
    BM25_B
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build index and docmap")
    tf_parser = subparsers.add_parser("tf", help="Print the term frequency for that term in the document with the given ID")
    tf_parser.add_argument("doc_id", type=int, help="doc_id for frequencey check")
    tf_parser.add_argument("term", type=str, help="string to frequency check")
    idf_parser = subparsers.add_parser("idf", help="Create inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to check IDF")
    tfidf_parser = subparsers.add_parser("tfidf", help="Generate TFIDF for term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID for TFIDF")
    tfidf_parser.add_argument("term", type=str, help="term for TFIDF")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{res["id"]}. {res['title']}")
        case "build":
            build_command()
        case "tf":
            print(f"Checking for frequency of {args.term} in Document:{args.doc_id}")
            count = tf_command(args.doc_id, args.term)
            print(f"{args.term} appears in Document:{args.doc_id} {count} times.")
        case "idf":
            print(f"Check inverse document frequency for term: {args.term}")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            print(f"Generating TF-IDF score of '{args.term}' in document '{args.doc_id}'")
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            print(f"Generating BM25 IDF score of '{args.term}'")
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
    
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
