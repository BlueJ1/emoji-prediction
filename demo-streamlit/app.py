from enum import Enum
import pickle as pkl
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel
from pathlib import Path
try:
    from emoji_prediction.models import four_gram
    from emoji_prediction.models.four_gram import FourGram
    from emoji_prediction.models import mlp_unified
    from emoji_prediction.models import classic_ml_models_api
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.join(
        os.path.join(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..'), 'emoji_prediction'), 'models'))
    import four_gram
    from four_gram import FourGram
    import mlp_unified
    import classic_ml_models_api


class Model(Enum):
    FOURGRAM = 'Four-gram'
    # ONEGRAM = 'One-gram'
    MLPCONCAT = 'MLP with concatenation of embeddings'
    # MLPSUM = 'MLP with sum of embeddings'
    LOGREG = 'Logistic Regression'
    NAIVEBAYES = 'Naive Bayes'


class ModelInput(BaseModel):
    text: str
    chosen_model: Model
    index: int


def valid_index(index: int, text: str) -> bool:
    array_words = text.split()
    if index < 0 or index > len(array_words):
        return False


def add_prediction(prediction: str, text: str, index: int) -> str:
    array_words = text.split()
    array_words.insert(index, prediction)
    return " ".join(array_words)


def predict(text: str, chosen_model: str, index: int) -> str:
    print(chosen_model == Model.FOURGRAM)
    prediction = ""
    if chosen_model == Model.FOURGRAM.value:
        prediction = four_gram.four_gram_api_predict(text, index)

    elif chosen_model == "One-gram":
        print("Not implemented yet")

    elif chosen_model == Model.MLPCONCAT.value:
        prediction = mlp_unified.mlp_concat_api_predict(text, index)

    elif chosen_model == "MLP with sum of embeddings":
        print("Not implemented yet")

    elif chosen_model == Model.LOGREG.value:
        prediction = classic_ml_models_api.predict("log_reg", text)

    elif chosen_model == Model.NAIVEBAYES.value:
        prediction = classic_ml_models_api.predict("naive_bayes", text)

    return prediction


def main():
    st.title("Emoji Predictor :sparkles:")

    with st.form(key="pydantic_form"):
        data = sp.pydantic_input(key="my_input_model", model=ModelInput)
        chosen_model = st.selectbox("Choose a model", [model.value for model in Model])
        data["chosen_model"] = chosen_model
        print(data)
        st.write(data)
        if valid_index(data["index"], data["text"]) is False:
            st.warning("Please enter a valid index")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if data == "":
            st.warning("You did not enter a tweet yet")
        else:
            prediction = predict(data["text"],
                                 data["chosen_model"], data["index"])
            st.write(add_prediction(prediction, data["text"], data["index"]))


if __name__ == '__main__':
    main()
