from dotenv import load_dotenv
from fastapi import FastAPI

from retrieval import run_retrieval

load_dotenv()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Up and Running"}


@app.get("/retrieval")
async def rag(query: str):
    return run_retrieval(query).response
