from pydantic import BaseModel, EmailStr

# Pydantic model for request validation
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChatHistory(BaseModel):
    user_id: str

class ChatFlow(BaseModel):
    chat_name: str
    