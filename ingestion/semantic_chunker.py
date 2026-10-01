from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from dotenv import load_dotenv

load_dotenv()

def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path.

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.

    Args:
        markdown_file: Markdown file path.
        embeddings: Azure OpenAI embedding model.

    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    return splitter.create_documents([markdown_content])

if __name__ == "__main__":
    import os
    from langchain_openai import AzureOpenAIEmbeddings
    from pydantic import SecretStr

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key_value = os.getenv("AZURE_OPENAI_API_KEY")
    api_key = SecretStr(api_key_value) if api_key_value is not None else None
    api_version = os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION", "").strip() or "2023-05-15"
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env."
        )
    if not embedding_model or not embedding_model.strip():
        raise RuntimeError(
            "Missing Azure OpenAI embedding deployment. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env."
        )

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_deployment=embedding_model.strip(),
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    markdown_file = "../data/markdown/2024_Tesla.md"

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()