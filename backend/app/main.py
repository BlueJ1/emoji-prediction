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
        return {"prediction": prediction[0]}
    else:
        return {"There was no model loading despite "
                "having chosen a valid model"}
