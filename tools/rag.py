class RAG:
    def __init__(self, file_store):
        self.file_store = file_store

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Returns a list of text chunks from the file store most relevant to query.
        # Read-only: writing is handled by knowledge/update_gate.py.
        return self.file_store.retrieve(query, top_k=top_k)
