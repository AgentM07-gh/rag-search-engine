from lib.tokenization import token_split

class InvertedIndex:

    def __init__(self, index, docmap):
        self.index: dict[str, set[int]]
        self.docmap: dict[int, object]

    def __add_document(self, doc_id, text):
        token_list = token_split(text)
        for token in token_list:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}
    
    def get_documents(self, term)->list:
        doc_ids = list(self.index.get(term, set()))
        sorted_doc_ids = sorted(doc_ids)
        return sorted_doc_ids