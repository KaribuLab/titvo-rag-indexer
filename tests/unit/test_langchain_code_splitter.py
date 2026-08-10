from rag_indexer.infra.adapters.langchain_code_splitter import LangChainCodeSplitter


class TestLangChainCodeSplitter:
    def test_split_python_file(self):
        splitter = LangChainCodeSplitter(chunk_size=100, chunk_overlap=20)
        content = """
def hello_world():
    print("Hello, World!")
    return 42

class MyClass:
    def method1(self):
        return 1

    def method2(self):
        return 2
"""
        chunks = splitter.split("src/main.py", content)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_javascript_file(self):
        splitter = LangChainCodeSplitter(chunk_size=100, chunk_overlap=20)
        content = """
function hello() {
    console.log("Hello");
}

const x = 42;
"""
        chunks = splitter.split("src/app.js", content)

        assert len(chunks) > 0

    def test_excluded_extensions_return_empty(self):
        splitter = LangChainCodeSplitter()

        # Binary files should return empty list
        assert splitter.split("image.png", "binary content") == []
        assert splitter.split("archive.zip", "binary content") == []
        assert splitter.split("data.db", "binary content") == []

    def test_excluded_directories_return_empty(self):
        splitter = LangChainCodeSplitter()

        # Files in excluded directories should return empty list
        assert splitter.split("node_modules/package/index.js", "content") == []
        assert splitter.split(".git/config", "content") == []
        assert splitter.split("__pycache__/module.pyc", "content") == []

    def test_unsupported_extension_uses_generic_splitter(self):
        splitter = LangChainCodeSplitter(chunk_size=50, chunk_overlap=10)
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

        # Files without recognized extension still get split
        chunks = splitter.split("readme.txt", content)
        assert len(chunks) > 0

    def test_empty_content(self):
        splitter = LangChainCodeSplitter()
        chunks = splitter.split("empty.py", "")
        assert chunks == []
