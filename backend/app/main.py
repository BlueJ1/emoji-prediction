from pydantic import BaseModel
import pickle as pkl
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse


class ModelInput(BaseModel):
    text: str
    chosen_model: str


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


@app.post("/predict", tags=["emoji prediction API"])
async def predict_emoji(data: ModelInput):
    """
    Predicts emoji based on text
    """
    if data["text"] == "":
        raise HTTPException(status_code=400,
                            detail="Please enter a text to predict emoji")
    if data["chosen_model"] == "":
        raise HTTPException(status_code=400,
                            detail="Please choose a model to predict emoji")
    if data["chosen_model"] == "Four-gram":
        model = pkl.load(open("models/four_gram_model.pkl", "rb"))
    elif data["chosen_model"] == "One-gram":
        model = pkl.load(open("models/one_gram_model.pkl", "rb"))
    elif data["chosen_model"] == "MLP":
        model = pkl.load(open("models/mlp_model.pkl", "rb"))
    else:
        raise HTTPException(status_code=400,
                            detail="Please choose a valid model")
    prediction = model.predict([data["text"]])
    return {"prediction": prediction[0]}
