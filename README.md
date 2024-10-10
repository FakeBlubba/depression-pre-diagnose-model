# Depression Analysis Classifier

## Overview
The **Depression Analysis Classifier** is a machine learning tool designed to predict signs of depression from text data. It uses Transformer-based models, such as BERT, to analyze language and classify texts, with the goal of contributing to mental health initiatives by offering an automated method to detect potential symptoms of depression.

## Features
- **BERT-Based Classification**: The system relies on the BERT encoder to understand and classify text, taking advantage of modern advances in pre-trained language models.
- **Customizable Hyperparameters**: Allows adjustment of key parameters such as learning rate to improve model performance.
- **Integration with Audio Transcription**: Includes the ability to transcribe audio responses for text analysis if needed.

## Installation
To set up the project, follow these steps:

### Prerequisites
1. Make sure you have **Python 3.6+** installed on your system.
2. Install the dependencies specified in the `requirements.txt` file with the command:

   ```bash
   pip install -r requirements.txt
   ```
3. If you want to use a configuration with a GPU you have to check on the pytorch site how to do it properly given the different versions

Otherwise, for a configuration without GPU, use the following command:

```bash
pip3 install torch torchvision torchaudio
```

If it doesn't work you should probably check on https://pytorch.org/get-started/locally/.