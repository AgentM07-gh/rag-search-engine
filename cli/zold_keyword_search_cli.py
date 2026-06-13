#!/usr/bin/env python3

import argparse
from lib.keyword_dict_search import keyword_movie_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
             print("Searching for:", args.query)
             pass
        case _:
            parser.print_help()
    
    movie_list = keyword_movie_search(args.query)
    for movie in movie_list:
        movie_num = movie_list.index(movie) + 1
        print(f"{movie_num}: {movie}")

if __name__ == "__main__":
    main()