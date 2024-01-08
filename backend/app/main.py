
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from starlette.responses import RedirectResponse

# from .config.settings import settings
# from .config.database import engine, Base
# Creating the database connection
# Base.metadata.create_all(bind=engine)

tags_metadata = [
    {
        "name": "emoji prediction API",
        "description": "API to predict emojis based on text",
        "externalDocs": {
            "description": "Emoji Prediction external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    }
]
app = FastAPI()

# if settings.BACKEND_CORS_ORIGINS:
# Defines which domains are allowed to access the API.
# In production, be more strict.
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def home():
    return RedirectResponse(url="/docs")
