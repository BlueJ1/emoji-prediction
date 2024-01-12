from pydantic import BaseModel


class Emoji(BaseModel):
    text: str
    emoji: str
