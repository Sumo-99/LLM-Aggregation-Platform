from pydantic import BaseModel, EmailStr

# Pydantic model for request validation
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str