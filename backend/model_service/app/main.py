from fastapi import FastAPI
from app.routes import test
from app.routes import gemini
from app.routes import openai

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the model service!"}

# Include routes
app.include_router(test.router)
app.include_router(gemini.router)
app.include_router(openai.router)