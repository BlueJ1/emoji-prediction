import streamlit as st


def main():
    st.title("Emoji Predictor :sparkles:")
    txt = st.text_area("Enter tweet to add emoji to",
                       "It was the best of times, it was the"
                       "worst of times, it was the age of "
                       "wisdom, it was the age of foolishness,"
                       "it was the epoch of belief, it was the"
                       "epoch of incredulity, it was the season"
                       "of Light, it was the season of Darkness,"
                       "it was the spring of hope, it was the winter of "
                       "despair, (...)",
                       max_chars=280,
                       help="Enter a text of maximum 280 characters "
                            "(tweet length)")
    st.write(f'You wrote {len(txt)} characters.')
    chosen_model = st.radio("Choose a model",
                            ("Four-gram", "One-gram", "MLP"),
                            help="Choose a model to predict emoji with")
    st.write(f'You chose {chosen_model}')
    if txt == "":
        st.warning("Please enter a text to predict emoji")
    if chosen_model == "":
        st.warning("Please choose a model to predict emoji")

    if st.button("Predict Emoji"):
        st.write("We have not implemented this yet :heart:")


if __name__ == '__main__':
    main()
