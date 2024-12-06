from fastapi import FastAPI
from app.routes import test
from app.routes import signup
from app.routes import login

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the app backend service!"}

# Include routes
app.include_router(test.router)
app.include_router(signup.router)
app.include_router(login.router)