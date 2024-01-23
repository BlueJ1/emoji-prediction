# Emoji Prediction

## Getting started

### Prerequisites

Please use Python 3.10

Install the requirements:

```bash
pip install -r requirements.txt
```

### Data

Download the data from [here](https://www.kaggle.com/datasets/rexhaif/emojifydata-en)
(train.txt and test.txt) and put it in the `emoji_prediction/data` folder.

And download the glove.6B.50d.txt embeddings file from [here](https://nlp.stanford.edu/projects/glove/).

### Preprocessing

Please run the following command to preprocess the data:

```bash
python3 emoji_prediction/data_preprocessing/preprocessing.py
```

### Run App

```bash
streamlit run demo-streamlit/app.py
```

### Run Api

Start the flask api:

```bash
python3 demo-streamlit/flask_api.py
```

Then, you can make api calls, e.g. via curl:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"How funny", "chosen_model": "Naive Bayes", "index": 2}' http://127.0.0.1:5000/predict
```

## Authors

- [Wojciech Anyszka]
- [Laura M. Quirós]
- [Lennart August]
