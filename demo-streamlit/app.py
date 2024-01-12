from enum import Enum
import pickle as pkl
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel
from emoji_prediction.models.four_gram import four_gram_api_predict


class Model(Enum):
    FOURGRAM = 'Four-gram'
    ONEGRAM = 'One-gram'
    MLPCONCAT = 'MLP with concatenation of embeddings'
    MLPSUM = 'MLP with sum of embeddings'


class ModelInput(BaseModel):
    text: str
    chosen_model: Model
    index: int


def valid_index(index: int, text: str) -> bool:
    array_words = text.split()
    if index < 0 or index >= len(array_words):
        return False


def add_prediction(prediction: str, text: str, index: int) -> str:
    array_words = text.split()
    array_words.insert(index, prediction)
    return " ".join(array_words)


def train(text: str, chosen_model: str, index: int) -> str:
    model = ""
    prediction = ""
    if chosen_model == "Four-gram":
        model = pkl.load(open("emoji_prediction/models/four_gram.pkl", "rb"))
        print("This is the model: ", model)
        prediction = four_gram_api_predict(text, index)
    """
    elif chosen_model == "One-gram":
        model = pkl.load(open("models/one_gram_model.pkl", "rb"))
    elif chosen_model == "MLPConcat":
        model = pkl.load(open("models/concat_model.pkl", "rb"))
    elif chosen_model == "MLPSum":
        model = pkl.load(open("models/sum_model.pkl", "rb"))
    prediction = model.predict(text, index)
    """
    return prediction


def main():
    st.title("Emoji Predictor :sparkles:")

    with st.form(key="pydantic_form"):
        data = sp.pydantic_input(key="my_input_model", model=ModelInput)
        st.write(data)
        if valid_index(data["index"], data["text"]) is False:
            st.warning("Please enter a valid index")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if data == "":
            st.warning("You did not enter a tweet yet")
        else:
            prediction = train(data["text"],
                               data["chosen_model"], data["index"])
            st.write(add_prediction(prediction, data["text"], data["index"]))


if __name__ == '__main__':
    main()
