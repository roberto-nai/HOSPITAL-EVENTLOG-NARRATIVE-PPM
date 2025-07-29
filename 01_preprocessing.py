from pathlib import Path
import pandas as pd
import time
import json
from typing import List, Dict
import os
import sys
import logging
import warnings
from dotenv import load_dotenv
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from local_functions import ensure_dir_with_gitkeep
# Constants
# Input
EVENT_LOG_DIR = Path("event_log")
EVENT_LOG_FILE = "orbassano.csv"
OUTPUT_DIR = Path("event_log_narratives")
MAX_WORKERS = 3
# Output
OUTPUT_NARRATIVE_JSON_FILE = f"{Path(EVENT_LOG_FILE).stem}_narratives.json"
OUTPUT_NARRATIVE_CSV_FILE = f"{Path(EVENT_LOG_FILE).stem}_narratives.csv"
OUTPUT_OUTCOME_FILE = f"{Path(EVENT_LOG_FILE).stem}_outcomes.csv"
# CSV columns
OUTCOME_COL = "OUTCOME"
CASEID_COL = "CASEID"
TIMESTAMP_COL = "TIMESTAMP"
RESOURCE_COL = "RESOURCE"
ACTIVITY_COL = "ACTIVITY"
CSV_SEP = ";"

# Templates
OUTCOME_TEMPLATE = "The outcome for the patient is {{OUTCOME}}."

# Load environment variables and OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
LOG_FILE = Path(sys.argv[0]).with_suffix(".log")
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__} in {filename}:{lineno} - {message}")

warnings.showwarning = custom_warning_handler

def load_event_log(filepath: Path, csv_sep: str = CSV_SEP) -> pd.DataFrame:
    """
    Load the event log from the specified CSV file.
    """
    df = pd.read_csv(filepath_or_buffer=filepath, sep=csv_sep, parse_dates=["TIMESTAMP", "TIMESTAMP_END", "DATE"])
    df = df.sort_values(by=[CASEID_COL, "TIMESTAMP"]).reset_index(drop=True)
    return df

def build_event_log_string(group: pd.DataFrame) -> str:
    """
    Build a string representation of the event log for a single case.
    """
    lines = []
    for _, row in group.iterrows():
        line = (
            f"DAY: {row['DAY_start']}, RESOURCE: {row['RESOURCE']}, ACTIVITY: {row['ACTIVITY']}, "
            f"TIMESTAMP: {row['TIMESTAMP']}, ESI: {row['ESI']}, "
            f"CURRENT_ESI_1: {row['CURRENT_ESI_1']}, CURRENT_ESI_2: {row['CURRENT_ESI_2']}, "
            f"CURRENT_ESI_3: {row['CURRENT_ESI_3']}, CURRENT_ESI_4: {row['CURRENT_ESI_4']}, "
            f"CURRENT_ESI_5: {row['CURRENT_ESI_5']}, SHIFT: {row['SHIFT']}, OUTCOME: {row[OUTCOME_COL]}"
        )
        lines.append(line)
    return "\n".join(lines)

def gpt4o_generate_narrative(event_log: str, case_id=None) -> str:
    """
    Call OpenAI GPT-4o to generate a narrative from the event log string.
    """
    prompt = [
        {"role": "system",
         "content": (
             "You are a clinical report generator. Based on the following structured emergency department events, "
             "produce a narrative following this format:\n\n"
             "The day {{DAY}}, {{RESOURCE}} started performing {{ACTIVITY}} at {{TIMESTAMP}} on a patient. "
             "The patient was assigned with ESI {{ESI}}. "
             "At the moment of the activity, {{CURRENT_ESI_1}} patients had a ESI of level 1, "
             "{{CURRENT_ESI_2}} patients had a ESI of level 2, {{CURRENT_ESI_3}} patients had a ESI of level 3, "
             "{{CURRENT_ESI_4}} patients had a ESI of level 4, {{CURRENT_ESI_5}} patients had a ESI of level 5. "
             "The activity was performed during the {{SHIFT}} shift. "
             "The outcome for the patient is {{OUTCOME}}.\n\n"
             "Fill in the placeholders with values extracted from the data and write fluent English sentences."
         )
        },
        {"role": "user", "content": event_log}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in GPT-4o call for case {case_id}: {e}")
        return ""

def build_narratives_gpt(df: pd.DataFrame, max_workers: int = 3) -> List[Dict]:
    """
    Build a list of narratives using GPT-4o for each case, in parallel.
    """
    narratives = []
    grouped = list(df.groupby(CASEID_COL))

    def process_case(case_id, group):
        group = group.sort_values(by="TIMESTAMP")
        outcome = str(group.iloc[0][OUTCOME_COL])
        print("Case ID:", case_id, "Outcome:", outcome)
        event_log_str = build_event_log_string(group)
        narrative = gpt4o_generate_narrative(event_log_str, case_id=case_id)
        target_sentence = OUTCOME_TEMPLATE.replace("{{OUTCOME}}", outcome)
        return {
            "case_id": case_id,
            "narrative": narrative,
            "true_outcome": outcome,
            "target_sentence": target_sentence
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {executor.submit(process_case, case_id, group): case_id for case_id, group in grouped}
        for future in as_completed(future_to_case):
            case_id = future_to_case[future]
            try:
                result = future.result()
                narratives.append(result)
            except Exception as exc:
                logging.error(f"Exception for case {case_id}: {exc}")
    return narratives

def save_narratives_to_json(narratives: List[Dict], output_path: Path) -> None:
    """
    Save the list of narratives to a JSON file.
    """
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(narratives, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving narratives to JSON: {e}")

def save_narratives_to_csv(narratives: List[Dict], output_path: Path, csv_sep: str = CSV_SEP) -> None:
    """
    Save the list of narratives to a CSV file.
    """
    try:
        df_narratives = pd.DataFrame(narratives)
        df_narratives.to_csv(output_path, sep=csv_sep, index=False)
    except Exception as e:
        logging.error(f"Error saving narratives to CSV: {e}")

if __name__ == "__main__":
    print()
    print("*** PROGRAM START ***")
    print()

    start_time = time.time()

    # Output directory setup
    print("> Setting up output directory...")
    ensure_dir_with_gitkeep(OUTPUT_DIR)
    print(f"Output directory is set to: {OUTPUT_DIR}")
    print()

    # Load data
    print("> Loading event log...")
    log_path = EVENT_LOG_DIR / EVENT_LOG_FILE
    try:
        df_log = load_event_log(log_path)
    except Exception as e:
        logging.error(f"Error loading event log: {e}")
        sys.exit(1)

    print(f"Loaded event log from '{log_path}'.")
    print(f"Loaded event log with {len(df_log)} events and {df_log[CASEID_COL].nunique()} cases.")
    print(f"Columns: {', '.join(df_log.columns)}")
    print(f"First 5 rows:\n{df_log.head()}")
    print() 
    print("Distinct values in OUTCOME_COL:", list(df_log[OUTCOME_COL].unique()))
    # Calculation of absolute value and percentage of distinct “OUTCOME” values by CASEID
    caseid_counts = df_log.groupby(OUTCOME_COL)[CASEID_COL].nunique()
    caseid_percent = caseid_counts / caseid_counts.sum() * 100
    outcome_summary = pd.DataFrame({
        'count': caseid_counts,
        'percentage': caseid_percent.round(2)
    }).reset_index()
    outcome_summary = outcome_summary.sort_values(by='count', ascending=False).reset_index(drop=True)
    print("\n OUTCOME distribution (absolute and percentage):")
    print(outcome_summary)
    print("Saving outcome summary to CSV file...")
    path_outcome_summary = OUTPUT_DIR / OUTPUT_OUTCOME_FILE
    try:
        outcome_summary.to_csv(path_outcome_summary, sep=CSV_SEP, index=False)
        print(f"Outcome summary saved to '{path_outcome_summary}'")
    except Exception as e:
        logging.error(f"Error saving outcome summary: {e}")
    print()

    # Filter only cases with desired OUTCOME values
    print("> Filtering cases with desired outcomes...")
    allowed_outcomes = ["A domicilio", "Ricoverato"]
    df_log_clean = df_log[df_log[OUTCOME_COL].isin(allowed_outcomes)].reset_index(drop=True)
    print(f"Cases with outcome filtered: {df_log_clean[CASEID_COL].nunique()}")
    path_file_clean = OUTPUT_DIR / f"{Path(EVENT_LOG_FILE).stem}_filtered.csv"
    try:
        df_log_clean.to_csv(path_file_clean, sep=CSV_SEP, index=False)
        print(f"Saving cleaned event log to '{path_file_clean}'")
    except Exception as e:
        logging.error(f"Error saving cleaned event log: {e}")
    print()

    # Narrative data
    print(f"> Building narratives with GPT-4o (parallel, {MAX_WORKERS} at a time)...")
    narratives = build_narratives_gpt(df_log_clean, max_workers=MAX_WORKERS)
    print(f"Generated {len(narratives)} narratives.")
    print()

    # Save narratives to JSON and CSV files
    print("> Saving narratives to JSON and CSV files...")
    path_narrative_csv = OUTPUT_DIR / OUTPUT_NARRATIVE_CSV_FILE
    path_narrative_json = OUTPUT_DIR / OUTPUT_NARRATIVE_JSON_FILE
    save_narratives_to_json(narratives, path_narrative_json)
    save_narratives_to_csv(narratives, path_narrative_csv)
    print(f"Narratives saved to '{path_narrative_json}' and '{path_narrative_csv}'.")
    print()

    print("> Execution completed")
    end_time = time.time()
    delta_time = end_time - start_time
    print(f"Execution time: {delta_time:.2f} seconds")

    print()
    print("*** PROGRAM END ***")
    print()