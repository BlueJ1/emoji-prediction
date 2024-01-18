import json
import pickle as pkl

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse

try:
    from emoji_prediction.models import four_gram
    from emoji_prediction.models import mlp_unified
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.join(
        os.path.join(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../..'), 'emoji_prediction'), 'models'))
    import four_gram
    import mlp_unified

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
    if (int(index) < 0) or (index >= len(array_words)+1):
        return False
    return True


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

    def getText(self):
        return self.text
    def getModel(self):
        return self.chosen_model
    def getIndex(self):
        return self.index


@app.post("/predict", tags=["emoji prediction API"])
async def predict_emoji(data: ModelInput = None):
    """
    Predicts emoji based on text
    """
    model = ""

    # take values out of json model
    assert(isinstance(data, ModelInput))
    text = data.getText()
    assert(isinstance(text, str))
    model = data.getModel()
    assert(isinstance(model, str))
    index = data.getIndex()
    assert(isinstance(index, int))

    if text == "":
        raise HTTPException(status_code=400,
                            detail="Please enter a text to predict emoji")
    if valid_index(index, text) is False:
        raise HTTPException(status_code=400,
                            detail="Please enter a valid index")
    if model == "":
        raise HTTPException(status_code=400,
                            detail="Please choose a model to predict emoji")
    if model not in ["One-gram", "Four-gram", "MLPConcat", "MLPSum"]:
        raise HTTPException(status_code=400,
                            detail="Please choose a valid model")

    prediction = ""
    if model == "Four-gram":
        prediction = four_gram.four_gram_api_predict(text, index)
    elif model == "MLPConcat":
        prediction = mlp_unified.mlp_concat_api_predict(text, index)
    """
    elif chosen_model == "One-gram":
        model = pkl.load(open("emoji_prediction/models/one_gram.pkl", "rb"))
    
    elif chosen_model == "MLPSum":
        model = pkl.load(open("emoji_prediction/models/sum_model.pkl", "rb"))
    """
    if model != "":
        return add_prediction(prediction, text, index)
    else:
        print(f"Model was: {model}")
        return {"There was no model loading despite "
                "having chosen a valid model"}
