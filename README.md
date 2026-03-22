# NanoSocrates: Foundational Models, Building a Very Small Semantic Language Model

This project implements a  small foundational model from scratch named **NanoSocrates**,
 based on a T5-like Encoder-Decoder Transformer architecture.
---

## Project Structure

```text
NanoSocrates/
├── data/
│   ├── dataset.jsonl
│   └── special_tokens.txt
│
├── src/
│   ├── model.py
│   ├── tokenizer.py
│   ├── database_creator.py
│   └── utils.py
│
├── tokenizer/
│   ├── tokenizer.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── corpus.txt
│
├── weights/  
│
├── runs/
│   
├── requirements.txt
└── README.md
```
---

## Prerequisites

- Python 3.10 or higher  
- GPU with CUDA (recommended for training)
---

## Setup Instructions

1. **Create and activate a virtual environment**  
   Run: `python -m venv .venv`  
   Then: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)

2. **Install dependencies**  
   Run: `pip install -r requirements.txt`

---

## Dataset Preparation

The dataset used for training and validation is stored in `data/dataset.jsonl`.  
To rebuild it, run: `python src/database_creator.py`  
This script merges and cleans source files, then saves a unified dataset and the list of special tokens required by the tokenizer.

---

## Tokenizer Creation

(Re)train or adjust the tokenizer by running: `python src/tokenizer.py`  
Tokenizer artifacts will be saved in the `tokenizer/` directory.

---

## Training

From the project root (`NanoSocrates/`), start the training with:

`
python src/model.py
`

All the hyperparameters can be set inside the get_config().

--- 

## Results

The results are available in the `results/` folder, outside this project. Due to time and hardware constraints, 
a full validation was performed only every 20 epochs. 
The intermediate validation results are from a faster, indicative check used for monitoring purposes during training.

---
Detailed analysis is provided in the report:  
`NanoSocrates_report_Gentile.pdf`
