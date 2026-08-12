"""Tests for LangChainCodeSplitter.iter_chunks."""

from rag_indexer.domain.dto.file_content_dto import FileContent
from rag_indexer.infra.adapters.langchain_code_splitter import LangChainCodeSplitter


class TestIterChunks:
    def test_5C_2_iter_chunks_returns_iterator(self):
        from rag_indexer.infra.adapters.langchain_code_splitter import (
            LangChainCodeSplitter,
        )

        splitter = LangChainCodeSplitter(chunk_size=100, chunk_overlap=10)
        file = FileContent(
            path="src/main.py",
            content="def hello(): pass\n" * 20,
        )
        result = splitter.iter_chunks(file)
        # Returns an iterator (generator is OK)

        assert hasattr(result, "__iter__")
        chunks = list(result)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_5C_3_iter_chunks_empty_file(self):
        splitter = LangChainCodeSplitter(chunk_size=100, chunk_overlap=10)
        file = FileContent(path="src/main.py", content="")
        chunks = list(splitter.iter_chunks(file))
        assert chunks == []
