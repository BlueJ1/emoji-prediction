from enum import Enum
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel


class Model(Enum):
    FOURGRAM = 'Four-gram'
    ONEGRAM = 'One-gram'
    MLPCONCAT = 'MLP with concatenation of embeddings'
    MLPSUM = 'MLP with sum of embeddings'


class ModelInput(BaseModel):
    text: str
    chosen_model: Model
    index: int


def train(text: str, model: str, index: int) -> str:
    return "😀"


def main():
    st.title("Emoji Predictor :sparkles:")

    with st.form(key="pydantic_form"):
        data = sp.pydantic_input(key="my_input_model", model=ModelInput)
        st.write(data)
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if data == "":
            st.warning("You did not enter a tweet yet")
        else:
            prediction = train(data["text"],
                               data["chosen_model"], data["index"])
            st.write(f"Predicted emoji: {prediction}")


if __name__ == '__main__':
    main()
