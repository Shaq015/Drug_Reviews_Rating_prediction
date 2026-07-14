#!/usr/bin/env python3
"""
Run local GenAI generation for the Drug Reviews final output files.

Goal:
- Load a local instruction-tuned LLM, default Qwen2.5-7B-Instruct.
- For each test review, generate:
  1. generated summary, max 10 words, complete and not truncated
  2. identified sentiment, positive or negative
- Save progress to CSV so the job can resume after disconnections/failures.
- Avoid writing XLSX during the long run. XLSX can be created later from the final CSV.

Designed for Slurm/SBATCH execution on a university cluster.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Hide pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Constants

REQUIRED_FINAL_COLUMNS = [
    "drug name",
    "condition",
    "review",
    "rating",
    "predicted rating",
    "generated summary",
    "identified sentiment",
]

DEBUG_COLUMNS = [
    "genai_model",
    "genai_prompt_style",
    "genai_parse_success",
    "genai_raw_output",
    "genai_repaired_from_raw_output",
    "genai_error",
]

VALID_SENTIMENTS = {"positive", "negative"}

# Summaries ending with these words often look truncated, mark them invalid and retry
BAD_SUMMARY_ENDINGS = {
    "and", "or", "but", "with", "without", "to", "of", "for", "from",
    "because", "due", "causing", "caused", "lack", "though", "although",
    "while", "after", "before", "in", "on", "at", "by", "than", "per",
    "nor", "into", "under", "over", "during",
}

BAD_SUMMARY_ENDING_PHRASES = {
    "causing weight", "without causing weight", "due to", "because of",
    "lack of", "side effects from", "helped with", "struggled with",
    "suffering from", "improved with", "worse after", "better after",
}


# Utility functions

def log(message: str) -> None:
    """Print a timestamped log message and flush immediately."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def setup_cache(cache_dir: str) -> Path:
    """Configure Hugging Face cache directories inside the project folder."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_path)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(cache_path)

    log(f"HF cache directory: {cache_path.resolve()}")
    return cache_path


def clean_memory() -> None:
    """Clear Python and CUDA memory before loading the local LLM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def print_gpu_info() -> None:
    """Print GPU information for debugging in the Slurm output log."""
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            log(f"GPU memory free: {free_mem / 1024**3:.2f} GB")
            log(f"GPU memory total: {total_mem / 1024**3:.2f} GB")
        except Exception as exc:
            log(f"Could not read GPU memory info: {exc}")


def safe_filename(name: str) -> str:
    """Convert a descriptive name into a safe filename."""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace(":", "_")
    )


def word_count(text: str) -> int:
    """Count words in a simple and robust way."""
    return len(str(text).strip().split())


def normalize_summary(summary: str) -> str:
    """Clean generated summary text without truncating it."""
    summary = str(summary).strip()
    summary = re.sub(r"\s+", " ", summary)
    summary = summary.strip(' \t\n\r"')
    return summary


def is_complete_summary(summary: str, max_words: int = 10) -> bool:
    """
    Validate that the summary is short and appears complete.
    """
    summary = normalize_summary(summary)

    if not summary:
        return False

    if word_count(summary) > max_words:
        return False

    # Avoid summaries that visibly end mid-thought
    if summary.endswith((",", ";", ":", "-", "—", "–", "&", "...", "…")):
        return False

    words = summary.split()
    last_word = re.sub(r"[^A-Za-z]", "", words[-1]).lower()

    if last_word in BAD_SUMMARY_ENDINGS:
        return False

    summary_lower = re.sub(r"\s+", " ", summary.lower()).strip(" .,!?:;")

    for bad_phrase in BAD_SUMMARY_ENDING_PHRASES:
        if summary_lower.endswith(bad_phrase):
            return False

    return True


def normalize_sentiment(sentiment: str) -> str:
    """Normalize sentiment text into exactly positive/negative when possible."""
    sentiment = str(sentiment).strip().lower()

    if "positive" in sentiment:
        return "positive"
    if "negative" in sentiment:
        return "negative"

    # Do not accept neutral, mixed, unsure, etc. The assignment requires binary sentiment
    return ""


# Prompt engineering

def build_prompt(row: pd.Series, prompt_style: str) -> str:
    """
    Build one of the selected prompt-engineering alternatives.

    Important design choice:
    - The original rating is NOT used in the prompt.
    - Sentiment is inferred from the review text itself.
    - Drug and condition are used only as context.
    """
    drug = str(row.get("drug name", ""))
    condition = str(row.get("condition", ""))
    review = str(row.get("review", ""))

    if prompt_style == "direct":
        return f"""
        You are analyzing patient drug reviews.

        Task:
        Generate exactly two fields:
        1. "generated_summary": a complete summary of the review in 10 words or fewer.
        2. "identified_sentiment": either "positive" or "negative".

        Rules:
        - Use only the review text to infer sentiment.
        - Do not use rating numbers.
        - Do not return neutral.
        - If the review is mixed, choose the dominant patient experience.
        - Positive means the medication helped more than it harmed.
        - Negative means side effects, lack of effect, or discontinuation dominate.
        - The summary must be a complete phrase, not cut off.
        - Return only valid JSON. No markdown. No explanation.

        Required JSON format:
        {{
        "generated_summary": "...",
        "identified_sentiment": "positive"
        }}

        Review:
        {review}
        """.strip()

    if prompt_style == "medical_focus":
        return f"""
        You analyze patient experiences with medications.

        Use the drug and condition only as context.
        Do NOT use any rating number to decide sentiment.

        Your task:
        1. Write a complete summary of the patient's experience in 10 words or fewer.
        2. Classify the expressed sentiment as exactly "positive" or "negative".

        Sentiment rules:
        - Choose "positive" if benefits, effectiveness, or improvement dominate.
        - Choose "negative" if side effects, lack of effect, worsening, or stopping treatment dominate.
        - If the review contains both benefits and side effects, choose the dominant overall experience.
        - Do not output "neutral", "mixed", or any other label.

        Summary rules:
        - The summary must be a complete phrase.
        - Do not end the summary with unfinished words such as "and", "or", "with", "without", "to", "of", "for", "because", or "due".
        - Do not mention rating numbers.
        - Do not copy long text from the review.
        - Maximum length: 10 words.

        Return only valid JSON:
        {{
        "generated_summary": "...",
        "identified_sentiment": "positive"
        }}

        Drug: {drug}
        Condition: {condition}
        Review:
        {review}
        """.strip()

    if prompt_style == "few_shot":
        return f"""
        You analyze patient drug reviews.

        Return only valid JSON with:
        {{
        "generated_summary": "...",
        "identified_sentiment": "positive"
        }}

        Rules:
        - The summary must be complete and 10 words or fewer.
        - Sentiment must be exactly "positive" or "negative".
        - Do not return neutral.
        - Use only the review text, drug, and condition.
        - Do not use rating numbers.

        Examples:

        Review: "This medicine helped my anxiety within two weeks. No major side effects."
        Output:
        {{"generated_summary": "Helped anxiety with no major side effects", "identified_sentiment": "positive"}}

        Review: "Terrible nausea and dizziness. I had to stop taking it."
        Output:
        {{"generated_summary": "Stopped due to nausea and dizziness", "identified_sentiment": "negative"}}

        Review: "It reduced headaches, but caused fatigue and nausea."
        Output:
        {{"generated_summary": "Reduced headaches but caused fatigue and nausea", "identified_sentiment": "negative"}}

        Now analyze this review.

        Drug: {drug}
        Condition: {condition}
        Review:
        {review}
        """.strip()

    raise ValueError(
        f"Unknown prompt_style='{prompt_style}'. "
        "Use one of: direct, medical_focus, few_shot."
    )


def build_repair_prompt(raw_output: str, original_review: str) -> str:
    """
    Repair invalid GenAI output.

    This is used when:
    - JSON parsing failed.
    - The summary is longer than 10 words.
    - The summary appears incomplete.
    - Sentiment is not exactly positive/negative.
    """
    return f"""
    Fix the following model output.

    Return only valid JSON with exactly these fields:
    {{
    "generated_summary": "...",
    "identified_sentiment": "positive"
    }}

    Rules:
    - "generated_summary" must be a complete phrase.
    - Maximum summary length: 10 words.
    - Do not cut the sentence in the middle.
    - Do not end with unfinished words like "and", "or", "with", "without", "to", "of", "for", "because", or "due".
    - "identified_sentiment" must be exactly "positive" or "negative".
    - Do not output neutral or mixed.
    - Use the review text to decide sentiment.
    - Return JSON only. No explanation.

    Original review:
    {original_review}

    Invalid model output:
    {raw_output}
    """.strip()


# JSON parsing and validation

def parse_genai_output(raw_text: str) -> Dict[str, object]:
    """
    Parse the LLM output into the two required fields.

    Important: we do NOT truncate summaries to 10 words. If the summary is too
    long or incomplete, parse_success=False and the model gets one repair attempt.
    """
    raw_text = str(raw_text).strip()
    parsed = None

    # Attempt 1: direct JSON parsing
    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = None

    # Attempt 2: extract first JSON-like object from text
    if parsed is None:
        match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None

    if parsed is None or not isinstance(parsed, dict):
        return {
            "generated_summary": "",
            "identified_sentiment": "",
            "parse_success": False,
            "raw_output": raw_text,
        }

    summary = normalize_summary(parsed.get("generated_summary", ""))
    sentiment = normalize_sentiment(parsed.get("identified_sentiment", ""))

    parse_success = (
        is_complete_summary(summary, max_words=10)
        and sentiment in VALID_SENTIMENTS
    )

    if not parse_success:
        # Keep invalid fields empty
        return {
            "generated_summary": "" if not is_complete_summary(summary, 10) else summary,
            "identified_sentiment": sentiment if sentiment in VALID_SENTIMENTS else "",
            "parse_success": False,
            "raw_output": raw_text,
        }

    return {
        "generated_summary": summary,
        "identified_sentiment": sentiment,
        "parse_success": True,
        "raw_output": raw_text,
    }


# Model loading and generation

def load_qwen_model(model_path: str, use_4bit: bool, local_files_only: bool):
    """Load Qwen locally. Default is 4-bit to avoid memory crashes."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. Download it first or pass --allow_download."
        )

    clean_memory()
    print_gpu_info()

    log(f"Loading tokenizer from: {model_path.resolve()}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        log("Loading model in 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
    else:
        log("Loading model in FP16. Use this only if memory is sufficient.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )

    model.eval()
    log("GenAI model loaded successfully.")
    print_gpu_info()

    return tokenizer, model


def generate_with_model(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Generate a short response from the local instruction model."""
    messages = [
        {
            "role": "system",
            "content": (
                "Follow formatting instructions exactly. Return valid JSON only. "
                "Do not add explanations. Do not use neutral sentiment."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def generate_and_parse(
    tokenizer,
    model,
    row: pd.Series,
    prompt_style: str,
    max_new_tokens: int,
    temperature: float,
    retry_on_parse_fail: bool,
) -> Dict[str, object]:
    """Generate GenAI output for one row and parse it."""
    prompt = build_prompt(row, prompt_style=prompt_style)
    raw_output = generate_with_model(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    parsed = parse_genai_output(raw_output)

    # One repair attempt for invalid JSON, neutral sentiment, too-long summary, or truncated summary
    if retry_on_parse_fail and not parsed["parse_success"]:
        repair_prompt = build_repair_prompt(
            raw_output=raw_output,
            original_review=str(row.get("review", "")),
        )
        repaired_raw_output = generate_with_model(
            tokenizer=tokenizer,
            model=model,
            prompt=repair_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        repaired = parse_genai_output(repaired_raw_output)

        if repaired["parse_success"]:
            repaired["raw_output"] = repaired_raw_output
            repaired["repaired_from_raw_output"] = raw_output
            return repaired

    parsed["repaired_from_raw_output"] = ""
    return parsed


# File processing

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure final/debug columns exist and have safe dtypes for text assignment."""
    df = df.copy()

    for col in REQUIRED_FINAL_COLUMNS:
        if col not in df.columns:
            if col in {"generated summary", "identified sentiment"}:
                df[col] = ""
            else:
                raise ValueError(f"Missing required input column: {col}")

    for col in DEBUG_COLUMNS:
        if col not in df.columns:
            df[col] = False if col == "genai_parse_success" else ""

    text_columns = [
        "generated summary",
        "identified sentiment",
        "genai_model",
        "genai_prompt_style",
        "genai_raw_output",
        "genai_repaired_from_raw_output",
        "genai_error",
    ]
    for col in text_columns:
        df[col] = df[col].fillna("").astype("object")

    df["genai_parse_success"] = df["genai_parse_success"].fillna(False).astype(bool)

    return df


def row_needs_processing(row: pd.Series) -> bool:
    """Return True if the row is missing valid GenAI output."""
    summary = normalize_summary(row.get("generated summary", ""))
    sentiment = normalize_sentiment(row.get("identified sentiment", ""))

    if not is_complete_summary(summary, max_words=10):
        return True
    if sentiment not in VALID_SENTIMENTS:
        return True

    return False


def get_output_paths(output_dir: Path, alternative_name: str) -> Tuple[Path, Path]:
    """Return progress CSV and clean final CSV paths."""
    safe_alt_name = safe_filename(alternative_name)
    progress_csv_path = output_dir / f"{safe_alt_name}_progress_with_debug.csv"
    final_csv_path = output_dir / f"{safe_alt_name}_final.csv"
    return progress_csv_path, final_csv_path


def save_outputs(df: pd.DataFrame, progress_csv_path: Path, final_csv_path: Path) -> None:
    """
    Save progress and clean final CSV."""
    df.to_csv(progress_csv_path, index=False)
    final_df = df[REQUIRED_FINAL_COLUMNS].copy()
    final_df.to_csv(final_csv_path, index=False)


def process_one_file(
    tokenizer,
    model,
    input_path: Path,
    output_dir: Path,
    alternative_name: str,
    prompt_style: str,
    genai_model_name: str,
    max_rows: Optional[int],
    save_every: int,
    max_new_tokens: int,
    temperature: float,
    retry_on_parse_fail: bool,
) -> pd.DataFrame:
    """Process one final-output base file with one prompt style."""
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_csv_path, final_csv_path = get_output_paths(output_dir, alternative_name)

    # Resume logic
    if progress_csv_path.exists():
        log(f"Resuming from progress file: {progress_csv_path}")
        df = pd.read_csv(progress_csv_path)
    else:
        log(f"Starting from base file: {input_path}")
        df = pd.read_csv(input_path)

    df = ensure_output_columns(df)

    if max_rows is not None:
        process_indices = list(df.index[:max_rows])
        log(f"max_rows={max_rows}; processing only first {len(process_indices)} rows.")
    else:
        process_indices = list(df.index)
        log(f"Processing all rows: {len(process_indices)}")

    rows_to_process = [idx for idx in process_indices if row_needs_processing(df.loc[idx])]
    log(f"Rows needing GenAI output for {alternative_name}: {len(rows_to_process)}")

    processed_since_save = 0
    start_time = time.time()

    for counter, idx in enumerate(rows_to_process, start=1):
        try:
            parsed = generate_and_parse(
                tokenizer=tokenizer,
                model=model,
                row=df.loc[idx],
                prompt_style=prompt_style,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                retry_on_parse_fail=retry_on_parse_fail,
            )

            df.at[idx, "generated summary"] = parsed["generated_summary"]
            df.at[idx, "identified sentiment"] = parsed["identified_sentiment"]
            df.at[idx, "genai_model"] = genai_model_name
            df.at[idx, "genai_prompt_style"] = prompt_style
            df.at[idx, "genai_parse_success"] = bool(parsed["parse_success"])
            df.at[idx, "genai_raw_output"] = parsed.get("raw_output", "")
            df.at[idx, "genai_repaired_from_raw_output"] = parsed.get(
                "repaired_from_raw_output", ""
            )
            df.at[idx, "genai_error"] = ""

        except Exception as exc:
            # Keep the job alive, the row remains incomplete and can be retried later
            df.at[idx, "genai_error"] = repr(exc)
            log(f"Error at row index {idx}: {repr(exc)}")

        processed_since_save += 1

        if processed_since_save >= save_every:
            save_outputs(df, progress_csv_path, final_csv_path)
            elapsed_min = (time.time() - start_time) / 60
            log(
                f"Saved progress for {alternative_name}: "
                f"{counter}/{len(rows_to_process)} rows processed in this run, "
                f"elapsed={elapsed_min:.1f} min"
            )
            processed_since_save = 0

    save_outputs(df, progress_csv_path, final_csv_path)

    completed_mask = ~df.apply(row_needs_processing, axis=1)
    non_empty_mask = df["generated summary"].astype(str).str.strip().ne("")
    valid_sentiment_mask = df["identified sentiment"].astype(str).str.lower().isin(VALID_SENTIMENTS)

    log(f"Finished {alternative_name}")
    log(f"Progress CSV: {progress_csv_path}")
    log(f"Final CSV:    {final_csv_path}")
    log(f"Completed rows: {completed_mask.sum()} / {len(df)}")
    log(f"Rows with summary: {non_empty_mask.sum()} / {len(df)}")
    log(f"Rows with valid sentiment: {valid_sentiment_mask.sum()} / {len(df)}")

    return df


def rows_are_aligned(df1: pd.DataFrame, df2: pd.DataFrame, max_checks: int = 10) -> bool:
    """Check that the two alternative files refer to the same reviews in the same order."""
    if len(df1) != len(df2):
        return False

    check_cols = ["drug name", "condition", "review", "rating"]
    for col in check_cols:
        if col not in df1.columns or col not in df2.columns:
            return False

    # Check a few positions across the file, not every row, to keep this cheap
    positions = sorted(set([0, len(df1) - 1] + [int(i) for i in pd.Series(range(len(df1))).sample(min(max_checks, len(df1)), random_state=42)]))
    for pos in positions:
        for col in check_cols:
            if str(df1.iloc[pos][col]) != str(df2.iloc[pos][col]):
                return False

    return True


def create_second_alternative_from_shared_genai(
    source_df: pd.DataFrame,
    second_input_path: Path,
    output_dir: Path,
    second_alternative_name: str,
    genai_model_name: str,
    prompt_style: str,
) -> None:
    """
    Copy generated summary/sentiment from alternative 1 into alternative 2.

    This avoids generating the same summary twice. The two final files still differ
    in predicted rating because those columns come from their own rating models.
    """
    log("Shared GenAI mode enabled: creating alternative 2 without extra generation.")

    if (output_dir / f"{safe_filename(second_alternative_name)}_progress_with_debug.csv").exists():
        log("Existing alternative 2 progress file found; it will be overwritten in shared mode.")

    second_df = pd.read_csv(second_input_path)
    second_df = ensure_output_columns(second_df)

    if not rows_are_aligned(source_df, second_df):
        raise ValueError(
            "Alternative 1 and Alternative 2 files do not appear aligned. "
            "Cannot safely copy shared GenAI outputs. Run without --shared_genai."
        )

    copy_columns = [
        "generated summary",
        "identified sentiment",
        "genai_parse_success",
        "genai_raw_output",
        "genai_repaired_from_raw_output",
        "genai_error",
    ]
    for col in copy_columns:
        second_df[col] = source_df[col].values

    second_df["genai_model"] = genai_model_name
    second_df["genai_prompt_style"] = prompt_style

    progress_csv_path, final_csv_path = get_output_paths(output_dir, second_alternative_name)
    save_outputs(second_df, progress_csv_path, final_csv_path)

    completed_mask = ~second_df.apply(row_needs_processing, axis=1)
    log(f"Finished {second_alternative_name} from shared GenAI outputs")
    log(f"Progress CSV: {progress_csv_path}")
    log(f"Final CSV:    {final_csv_path}")
    log(f"Completed rows: {completed_mask.sum()} / {len(second_df)}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local Qwen GenAI summaries and sentiment for final drug review outputs."
    )

    parser.add_argument("--model_path", type=str, default="local_models/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--output_dir", type=str, default="final_genai_outputs")

    parser.add_argument(
        "--input_alt1",
        type=str,
        default="final_output_base_files/alternative_1_roberta_metadata_base.csv",
    )
    parser.add_argument(
        "--input_alt2",
        type=str,
        default="final_output_base_files/alternative_2_bert_base_base.csv",
    )

    parser.add_argument(
        "--prompt_alt1",
        type=str,
        default="direct",
        choices=["direct", "medical_focus", "few_shot"],
    )
    parser.add_argument(
        "--prompt_alt2",
        type=str,
        default="direct",
        choices=["direct", "medical_focus", "few_shot"],
    )

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--no_retry_on_parse_fail", action="store_true")
    parser.add_argument(
        "--only_alt",
        type=str,
        default="both",
        choices=["both", "alt1", "alt2"],
        help="Process both alternatives, only alt1, or only alt2.",
    )
    parser.add_argument(
        "--shared_genai",
        action="store_true",
        help=(
            "Generate summaries/sentiment once for alternative 1 and copy them to alternative 2. "
            "Use this when both alternatives use the same prompt and the same reviews."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_cache(args.cache_dir)

    input_alt1 = Path(args.input_alt1)
    input_alt2 = Path(args.input_alt2)
    output_dir = Path(args.output_dir)

    if args.only_alt in {"both", "alt1"} and not input_alt1.exists():
        raise FileNotFoundError(f"Alternative 1 input file not found: {input_alt1}")
    if args.only_alt in {"both", "alt2"} and not input_alt2.exists():
        raise FileNotFoundError(f"Alternative 2 input file not found: {input_alt2}")

    if args.shared_genai and args.only_alt != "both":
        raise ValueError("--shared_genai requires --only_alt both")
    if args.shared_genai and args.prompt_alt1 != args.prompt_alt2:
        raise ValueError("--shared_genai requires prompt_alt1 and prompt_alt2 to be identical")

    use_4bit = not args.no_4bit
    local_files_only = not args.allow_download
    retry_on_parse_fail = not args.no_retry_on_parse_fail

    tokenizer, model = load_qwen_model(
        model_path=args.model_path,
        use_4bit=use_4bit,
        local_files_only=local_files_only,
    )

    genai_model_name = Path(args.model_path).name

    alt1_df = None

    if args.only_alt in {"both", "alt1"}:
        alt1_df = process_one_file(
            tokenizer=tokenizer,
            model=model,
            input_path=input_alt1,
            output_dir=output_dir,
            alternative_name="alternative_1_roberta_metadata",
            prompt_style=args.prompt_alt1,
            genai_model_name=genai_model_name,
            max_rows=args.max_rows,
            save_every=args.save_every,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            retry_on_parse_fail=retry_on_parse_fail,
        )

    if args.shared_genai:
        create_second_alternative_from_shared_genai(
            source_df=alt1_df,
            second_input_path=input_alt2,
            output_dir=output_dir,
            second_alternative_name="alternative_2_bert_base",
            genai_model_name=genai_model_name,
            prompt_style=args.prompt_alt2,
        )
    elif args.only_alt in {"both", "alt2"}:
        process_one_file(
            tokenizer=tokenizer,
            model=model,
            input_path=input_alt2,
            output_dir=output_dir,
            alternative_name="alternative_2_bert_base",
            prompt_style=args.prompt_alt2,
            genai_model_name=genai_model_name,
            max_rows=args.max_rows,
            save_every=args.save_every,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            retry_on_parse_fail=retry_on_parse_fail,
        )

    log("All requested GenAI jobs completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(130)
