from dotenv import load_dotenv

from embed_knowledge_base import embed_knowledge_base
from retrieval import run_retrieval


def run_rag_pipeline():
    embed_knowledge_base()
    result = run_retrieval()
    print(result)


if __name__ == "__main__":
    load_dotenv()
    run_rag_pipeline()
