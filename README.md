# Depression Analysis Classifier
> [!IMPORTANT] This branch is not complete but it can give an overview of the try done using nltk libraries to capture language marks that could express depression.
## Overview
The Depression Analysis Classifier is a machine learning tool designed to predict depression from textual data. It utilizes linguistic features derived from WordNet, FrameNet, and sentiment analysis to perform classification. This project aims to contribute to mental health initiatives by providing an automated method to detect potential signs of depression in written text.

## Features
- Linguistic Analysis: Incorporates lexical resources like WordNet and FrameNet to extract meaningful linguistic patterns.
- Sentiment Analysis: Leverages sentiment analysis to gauge the emotional tone of texts.
- Customizable Weights: Allows fine-tuning the influence of different features on the prediction outcome.

## Installation
To set up this project, follow these steps:

### Prerequisites
1. Ensure you have Python 3.6+ installed on your system.

2. Then you have to install packages that you can find inside `requirements.txt` file. You can try copy and pasting this string inside your command line, otherwise you should see the package inside requirements and install it manually with `pip install` or `pip3 install `command.

   ```bash
   pip install -r requirements.txt
   ```

3. Then you should import and download all those packages:

   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('framenet_v17')
   nltk.download('punkt')
   ```

## Example of usage

You can read the following code to see an example of usage if we suppose you already have the transcription of your voice response to a question about depression:

```python
from classifier import DepressionAnalysisClassifier
classifier = DepressionAnalysisClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_new)
```

where `X_train` is the training set (it is preferred to use our `composite_dataset.csv` in the folder `/data`), `y_train` are the labels associated to `X_train`.

If you want to use the implementation of Text Transcription to answer some question with your microphone and try to predict a custom text, then you can try using methods inside `audio_manager.py` like this:

```python
import audio_manager
audio_manager.save_audio_file("record.wav")
X_new = audio_manager.transcription = audio_manager.transcript_audio("record.wav")
```

## Documentation

You can check the documentation by checking `/build/index.html`.

