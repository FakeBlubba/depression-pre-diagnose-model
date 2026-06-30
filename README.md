# Depression Pre-Diagnosis Model

NLP classifier for detecting signs of depression from text responses, built on BERT. Designed as a research tool to support mental health screening workflows.

---

## What it does

Given a text input — a written response or a transcribed audio answer — the model classifies whether it contains linguistic markers associated with depression, using a fine-tuned BERT encoder.

The system supports both text and audio input pipelines, making it applicable to interview-style screening contexts.

## Architecture

```
Text input / Audio transcription
        ↓
   Preprocessing & tokenization
        ↓
   BERT encoder (fine-tuned)
        ↓
   Binary classification head
        ↓
   Depression indicator score
```

## Features

- **BERT-based classification** — fine-tuned transformer encoder for mental health NLP
- **Audio transcription support** — converts spoken responses to text for analysis
- **Configurable hyperparameters** — learning rate, batch size, epochs adjustable via config
- **GPU/CPU compatible** — runs on both configurations

## Tech stack

- Python · PyTorch · Transformers (HuggingFace) · BERT
- Audio transcription pipeline
- scikit-learn · numpy

## Setup

**Requirements:** Python 3.6+

```bash
pip install -r requirements.txt
```

For GPU support, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your CUDA version.

For CPU only:
```bash
pip3 install torch torchvision torchaudio
```

## Usage

```bash
python main.py
```

Input can be provided as raw text or via the audio transcription module.

## Dataset & scope

This model is a research prototype intended for pre-diagnostic support — not a clinical tool. It should be used only as an exploratory aid, not as a substitute for professional mental health assessment.

## Branch

Active development is on the `BERT-VARIANT` branch.

---

**Topics:** `nlp` `bert` `mental-health` `classification` `pytorch` `transformers` `python`
