import pickle
from emoji import Emoji
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse

# create a FastAPI instance
app = FastAPI(
    title="Emoji Predictor",
    description="""
        This API endpoint predicts an emoji based on a given text.
        # Model usage
        You may use the API by sending a POST request to the
        /predict endpoint with a JSON body containing the text to predict.
        You also may choose the model to use by specifying the model name
        in the query string. The model name must be one of the following:
        `baseline` (), `four_gram`, `one-gram`, `mlp_sum`, `mlp_concat`.

        Example: `curl -X POST -H "Content-Type: application/json" -d
    '{"text": "I love python"}' http://localhost:8000/predict?model=baseline`

        # Limitations

    """,
    version="alpha",
)


@app.get("/", tags=["root"])
async def home():
    return RedirectResponse(url="/docs")


@app.get("/predict", response_model=Emoji, tags=["predict"])
async def predict(model: str):
    # load the model
    with open(f"models/{model}.pkl", "rb") as f:
        model = pickle.load(f)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    # make a prediction
    prediction = model.predict([[1, 2, 3, 4]])
    # return the prediction as a dictionary
    return Emoji(text="test", emoji=prediction[0])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
