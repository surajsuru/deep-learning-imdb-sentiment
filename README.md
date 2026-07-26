# IMDB Movie Review Sentiment Analysis

This project builds a simple deep learning web app for classifying movie reviews from the IMDB dataset as positive or negative.

## Overview

The app uses a pre-trained Simple RNN model saved as `simple_rnn_imdb.h5` and serves it through a Streamlit interface. Users can enter a review and get a prediction along with a confidence score.

## Features

- Predicts sentiment for user-entered movie reviews
- Uses a pre-trained Recurrent Neural Network (RNN) model
- Provides a simple interactive web interface with Streamlit

## Project Files

- `main.py` - Streamlit app for sentiment prediction
- `simple_rnn_imdb.h5` - Pre-trained model
- `IMDB Dataset.csv` - Dataset used for training
- `requirement.txt` - Python dependencies
- `prediction.ipynb` and `simpleRNN.ipynb` - Notebook versions for training and testing

## Requirements

Install the required packages using:

```bash
pip install -r requirement.txt
```

## How to Run

1. Activate your virtual environment (if you are using one)
2. Run the app:

```bash
streamlit run main.py
```

3. Open the local URL shown in the terminal in your browser

## Technology Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- scikit-learn

## Example

Enter a review such as:

> "This movie was amazing and very entertaining."

The app will classify it as positive or negative based on the trained model.

## Note

This project is a beginner-friendly demonstration of sentiment analysis using deep learning and a web interface.
