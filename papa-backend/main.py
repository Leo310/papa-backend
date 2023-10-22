from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

from retriever import run_retrieval
from synthesizer import run_synthesizer

load_dotenv()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Up and Running"}


@app.get("/retrieval")
async def retrieval(query: str):
    return run_retrieval(query)


@app.get("/synthesis")
async def synthesis(query: str):
    return await run_synthesizer(query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
