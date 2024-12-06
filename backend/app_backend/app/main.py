from fastapi import FastAPI
from app.routes import test
from app.routes import signup
from app.routes import login
from app.routes import model_fetch


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the app backend service!"}

# Include routes
app.include_router(test.router)
app.include_router(signup.router)
app.include_router(login.router)
app.include_router(model_fetch.router)

