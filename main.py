from fastapi import FastAPI
from pydantic import BaseModel

from rag import ask

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def query(q: Question):

    return ask(q.question)