from flask import Flask, request, jsonify
import app as streamlit_app  # Import the necessary functions from your Streamlit app

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    chosen_model = data['chosen_model']
    index = data['index']

    # Call the prediction function from your Streamlit app
    prediction = streamlit_app.predict(text, chosen_model, index)

    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run()
