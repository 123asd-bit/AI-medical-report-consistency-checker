# AI Medical Report Consistency Checker — Demo v1

A Streamlit prototype using Synthea synthetic CSV data.

## What works
- Patient selection
- Patient summary for conditions, allergies, medications and observations
- Chronological event timeline
- Transparent rule-based review checks:
  - Multiple values for the same observation on the same day
  - Allergy entries appearing with both active and ended status
  - Repeated active medication entries
- A clearly labelled illustrative contradiction example

## Important scope statement
This is a rule-based baseline prototype, not a trained clinical AI model. The illustrative allergy example uses a simple phrase rule and is not a model inference. The app is not a diagnostic tool and does not recommend treatment.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and the `data/` folder.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository, branch, and `app.py`.
5. Deploy.

## Dataset
The `data/` folder contains selected Synthea synthetic CSV files. Patient identifiers and demographics are synthetic. The app deliberately does not display direct identifiers such as SSN, passport or address.

## Next development steps
1. Add PDF ingestion with PyMuPDF and OCR fallback.
2. Add clinical NER using a task-appropriate fine-tuned checkpoint.
3. Add semantic candidate matching with Sentence Transformers.
4. Evaluate a clinical NLI model for contradiction classification.
5. Improve temporal handling, units, negation and clinical context.
6. Create an annotated patient-level evaluation set and compare rule-based, NLI and hybrid approaches.
