# NER‑Based Text Anonymization


510_global 2023

**Overview**  

A Python‑based pipeline that uses Hugging Face transformer models to detect personal names (and other PII) in textual data and replace them with generic placeholders.


## Why it matters

Humanitarian teams have privacy compliance. The code automates removal of personally identifiable information (PII) to meet GDPR, HIPAA, or internal data‑handling policies.  


## Setup

1. **Python ≥ 3.8** installed.  
2. Install the required packages  

   bash
   python -m pip install -r requirements.txt

## Usage

Two pretrained models are required:
**English** – dslim/bert-base-NER (trained on CoNLL‑2003)
**Spanish (cased)** – skimai/spanberta-base-cased-ner-conll02 (trained on CoNLL‑2002)

The scripts load the chosen model, run inference on a list of names (or any free‑text file), and export the results to inal.csv.

License MIT – feel free to adapt for your own humanitarian projects.
