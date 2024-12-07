from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    model_id: str
    session_id: str