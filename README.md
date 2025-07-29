# PPM_LLM_Outcome

This repository contains code and data for generating patient narratives from event logs and predicting patient outcomes using Large Language Models (LLMs).

## Project Structure

```
PPM_LLM_Outcome/
├── event_log/
│   └── orbassano.csv                # Raw event log data
├── output/
│   ├── orbassano_narratives.json    # Generated narratives (JSON)
│   ├── orbassano_narratives.csv     # Generated narratives (CSV)
│   ├── orbassano_predictions.json   # LLM predictions (JSON)
│   ├── llm_metrics.csv              # Evaluation metrics
│   └── orbassano_clean.csv          # Cleaned event log
├── 01_preprocessing.py              # Data preprocessing and narrative generation
├── 02_llm_prediction.py             # LLM-based outcome prediction and evaluation
├── local_functions.py               # Utility functions (e.g., ensure_dir_with_gitkeep)
└── README.md                        # Project documentation
```

## Requirements

- Python 3.8+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- pandas
- scikit-learn
- python-dotenv

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing and Narrative Generation

Run the following script to preprocess the event log and generate patient narratives:

```bash
python 01_preprocessing.py
```

- Loads the event log from `event_log/orbassano.csv`.
- Filters cases with allowed outcomes (`A domicilio`, `Ricoverato`).
- Generates a narrative for each patient case using OpenAI GPT-4o (calls are parallelised, 3 at a time).
- Handles errors and warnings with logging to a `.log` file.
- Ensures output directories exist and contain a `.gitkeep` file.
- Saves the narratives to `output/orbassano_narratives.json` and `output/orbassano_narratives.csv`.

### 2. LLM-based Outcome Prediction

Set your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Run the prediction script:

```bash
python 02_llm_prediction.py
```

- Loads narratives from `output/orbassano_narratives.json`.
- Uses OpenAI's GPT model to predict the outcome for each narrative.
- Saves predictions to `output/orbassano_predictions.json`.
- Evaluates predictions and saves metrics to `output/llm_metrics.csv`.

## Output Files

- **orbassano_clean.csv**: Cleaned event log with only allowed outcomes.
- **orbassano_narratives.json/csv**: Narratives for each patient case.
- **orbassano_predictions.json**: LLM-generated outcome predictions.
- **llm_metrics.csv**: Accuracy, precision, recall, and F1 score for the predictions.

## Notes

- The project is designed for research and experimentation with LLMs in medical outcome prediction.
- The event log and outputs are anonymised and for demonstration purposes only.
- The code assumes the event log contains specific columns (see `01_preprocessing.py`).
- Logging is enabled for errors and warnings.
- Narrative generation is parallelised for efficiency.

---
For questions or issues, please contact the project author roberto.nai@unito.it.
