import pickle as pkl
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse


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


@app.post("/predict-emoji", tags=["emoji prediction API"])
async def predict_emoji(text: str, chosen_model: str, index: int):
    """
    Predicts emoji based on text
    """
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
    if chosen_model == "Four-gram":
        model = pkl.load(open("models/four_gram_model.pkl", "rb"))
    elif chosen_model == "One-gram":
        model = pkl.load(open("models/one_gram_model.pkl", "rb"))
    elif chosen_model == "MLPConcat":
        model = pkl.load(open("models/concat_model.pkl", "rb"))
    elif chosen_model == "MLPSum":
        model = pkl.load(open("models/sum_model.pkl", "rb"))

    if model != "":
        prediction = model.predict(text, index)
        return add_prediction(prediction, text, index)
    else:
        return {"There was no model loading despite "
                "having chosen a valid model"}
