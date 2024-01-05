from typing import Optional

from pydantic import BaseModel, Field


class UserSubmissionDto(BaseModel):
    message: str
    
    
class BotResponseDto(BaseModel):
    data: Optional[str] = Field(None, description="Response data, if valid.")
    error: Optional[str] = Field(None, description="Response error, if invalid.")
