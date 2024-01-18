import pickle as pkl

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse
try:
    from emoji_prediction.models import four_gram
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(
        os.path.join(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../..'), 'emoji_prediction'), 'models'))
    import four_gram

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

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def valid_index(index: int, text: str) -> bool:
    array_words = text.split()
    if index < 0 or index >= len(array_words):
        return False


def add_prediction(prediction: str, text: str, index: int) -> str:
    array_words = text.split()
    array_words.insert(index, prediction)
    return " ".join(array_words)


@app.get("/", tags=["root"])
async def home():
    return RedirectResponse(url="/docs")


class ModelInput(BaseModel):
    text: str
    chosen_model: str
    index: int

@app.post("/predict", tags=["emoji prediction API"])
async def predict_emoji(model : ModelInput):
    """
    Predicts emoji based on text
    """
    text = model.text
    chosen_model = model.chosen_model
    index = model.index

    if text == "":
        raise HTTPException(status_code=400,
                            detail="Please enter a text to predict emoji")
    if valid_index(index, text) is False:
        raise HTTPException(status_code=400,
                            detail="Please enter a valid index")
    if chosen_model == "":
        raise HTTPException(status_code=400,
                            detail="Please choose a model to predict emoji")
    if chosen_model not in ["One-gram", "Four-gram", "MLPConcat", "MLPSum"]:
        raise HTTPException(status_code=400,
                            detail="Please choose a valid model")

    model = ""
    prediction = ""
    if chosen_model == "Four-gram":
        prediction = four_gram.four_gram_api_predict(text, index)
    """
    elif chosen_model == "One-gram":
        model = pkl.load(open("emoji_prediction/models/one_gram.pkl", "rb"))
    elif chosen_model == "MLPConcat":
        model = pkl.load(
        open("emoji_prediction/models/concat_model.pkl", "rb"))
    elif chosen_model == "MLPSum":
        model = pkl.load(open("emoji_prediction/models/sum_model.pkl", "rb"))
    """
    if model != "":
        return add_prediction(prediction, text, index)
    else:
        return {"There was no model loading despite "
                "having chosen a valid model"}
