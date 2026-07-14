#!/usr/bin/env python3
"""
Post-process GenAI output files after the full run.

Purpose:
- Do NOT rerun the full GenAI generation.
- Detect rows with missing summaries or non-English/CJK artifacts.
- Recover summaries from genai_raw_output when possible.
- Clean CJK artifacts when the English part is still valid.
- Save repaired final/progress files and a repair report.

Run from the project root:
    python repair_genai_outputs.py --output_dir final_genai_outputs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# CJK ranges catch Chinese/Japanese/Korean artifacts that appeared in a few summaries
CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")

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

VALID_SENTIMENTS = {"positive", "negative"}


def word_count(text: Any) -> int:
    """Count words in a robust way."""
    if pd.isna(text):
        return 0
    return len(str(text).strip().split())


def has_cjk(text: Any) -> bool:
    """Return True if the string contains Chinese/Japanese/Korean characters."""
    if pd.isna(text):
        return False
    return bool(CJK_PATTERN.search(str(text)))


def normalize_summary(summary: Any) -> str:
    """
    Clean a generated summary without mechanically truncating it.

    Important:
    - We remove CJK artifacts but keep the English part.
    - Do not cut the summary to 10 words here, because that can create incomplete summaries.
      The validation function decides if it is acceptable.
    """
    if pd.isna(summary):
        return ""

    summary = str(summary).strip()
    summary = summary.replace("\n", " ").replace("\r", " ")

    # Remove CJK artifacts if existing, but keep the English part
    summary = CJK_PATTERN.sub("", summary)

    # Remove common output prefixes if the model included them inside the value
    summary = re.sub(r"^(generated[_\s-]*summary|summary)\s*[:\-]\s*", "", summary, flags=re.I)

    # Normalize whitespace and quotes
    summary = re.sub(r"\s+", " ", summary)
    summary = summary.strip(" \t\n\r\"'")

    # Clean leftover punctuation created by removing foreign text
    summary = re.sub(r"\s*[:;]\s*", ": ", summary)
    summary = summary.strip(" :;,-")

    return summary


def normalize_sentiment(sentiment: Any) -> str:
    """Normalize sentiment to lowercase positive/negative when possible."""
    if pd.isna(sentiment):
        return ""
    s = str(sentiment).strip().lower()
    if "positive" in s and "negative" not in s:
        return "positive"
    if "negative" in s and "positive" not in s:
        return "negative"
    if s in VALID_SENTIMENTS:
        return s
    return s


def is_complete_summary(summary: Any, max_words: int = 10) -> bool:
    """Validate that a summary is non-empty, short, and does not look cut off."""
    summary = normalize_summary(summary)

    if not summary:
        return False
    if word_count(summary) > max_words:
        return False
    if has_cjk(summary):
        return False

    if summary.endswith((",", ";", ":", "-", "—", "–", "&", "...", "…")):
        return False

    last_word = re.sub(r"[^A-Za-z]", "", summary.split()[-1]).lower()
    if last_word in BAD_SUMMARY_ENDINGS:
        return False

    summary_lower = re.sub(r"\s+", " ", summary.lower()).strip(" .,!?:;")
    for phrase in BAD_SUMMARY_ENDING_PHRASES:
        if summary_lower.endswith(phrase):
            return False

    return True


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Try to extract a JSON object from raw model output.

    The debug file usually contains genai_raw_output. If the parser rejected the
    summary, the raw output may still contain a usable generated_summary field.
    """
    if not text or pd.isna(text):
        return None

    raw = str(text).strip()

    # First try direct JSON parsing
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Then try extracting the first {...} block.
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def extract_summary_from_raw(raw_output: Any) -> str:
    """Recover generated_summary from raw model output when possible."""
    if pd.isna(raw_output):
        return ""

    raw = str(raw_output)

    obj = extract_json_object(raw)
    if obj:
        for key in ["generated_summary", "generated summary", "summary"]:
            if key in obj:
                return normalize_summary(obj[key])

    # Regex fallback for imperfect JSON-like outputs
    patterns = [
        r'"generated_summary"\s*:\s*"([^"]+)"',
        r'"generated summary"\s*:\s*"([^"]+)"',
        r'"summary"\s*:\s*"([^"]+)"',
        r'generated_summary\s*[:=]\s*([^\n\r]+)',
        r'summary\s*[:=]\s*([^\n\r]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            value = match.group(1).strip().strip(",}")
            return normalize_summary(value)

    return ""


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV and fail with a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_repairs(progress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a row-level repair table from the debug/progress file.

    We use the progress file because it contains genai_raw_output, which is useful
    for recovering summaries that were rejected by the original parser.
    """
    repairs = []

    for idx, row in progress_df.iterrows():
        original_summary = row.get("generated summary", "")
        original_sentiment = row.get("identified sentiment", "")
        raw_output = row.get("genai_raw_output", "")

        cleaned_summary = normalize_summary(original_summary)
        sentiment = normalize_sentiment(original_sentiment)

        needs_repair = (
            not is_complete_summary(cleaned_summary)
            or has_cjk(original_summary)
            or sentiment not in VALID_SENTIMENTS
        )

        if not needs_repair:
            continue

        # First try recovering from the raw model output
        recovered_summary = extract_summary_from_raw(raw_output)
        recovered_summary = normalize_summary(recovered_summary)

        # If raw recovery fails but simple cleaning fixed the original summary, keep it
        if is_complete_summary(recovered_summary):
            final_summary = recovered_summary
            status = "recovered_from_raw_output"
        elif is_complete_summary(cleaned_summary):
            final_summary = cleaned_summary
            status = "cleaned_existing_summary"
        else:
            final_summary = ""
            status = "manual_review_required"

        repairs.append({
            "row_index": idx,
            "rating": row.get("rating", None),
            "drug name": row.get("drug name", row.get("drugName", None)),
            "condition": row.get("condition", None),
            "review": row.get("review", None),
            "original_summary": original_summary,
            "cleaned_summary": cleaned_summary,
            "recovered_summary": recovered_summary,
            "final_summary": final_summary,
            "original_sentiment": original_sentiment,
            "final_sentiment": sentiment,
            "repair_status": status,
            "raw_output": raw_output,
        })

    return pd.DataFrame(repairs)


def apply_repairs(df: pd.DataFrame, repairs: pd.DataFrame) -> pd.DataFrame:
    """Apply resolved summary repairs to a dataframe by row index."""
    fixed = df.copy()

    for _, repair in repairs.iterrows():
        idx = int(repair["row_index"])
        if idx >= len(fixed):
            continue

        if repair["repair_status"] != "manual_review_required":
            fixed.at[idx, "generated summary"] = repair["final_summary"]
            fixed.at[idx, "identified sentiment"] = repair["final_sentiment"]
            if "genai_parse_success" in fixed.columns:
                fixed.at[idx, "genai_parse_success"] = True

    return fixed


def summarize_quality(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Return key quality diagnostics for a final/progress dataframe."""
    summaries = df["generated summary"] if "generated summary" in df.columns else pd.Series(dtype=str)
    sentiments = df["identified sentiment"].map(normalize_sentiment) if "identified sentiment" in df.columns else pd.Series(dtype=str)

    missing_summary = summaries.isna() | (summaries.astype(str).str.strip() == "")
    over_10 = summaries.astype(str).str.split().str.len() > 10
    cjk = summaries.map(has_cjk)
    invalid_sentiment = ~sentiments.isin(VALID_SENTIMENTS)
    incomplete = ~summaries.map(is_complete_summary)

    return {
        "file": name,
        "rows": len(df),
        "missing_summary": int(missing_summary.sum()),
        "summaries_over_10_words": int(over_10.sum()),
        "summaries_with_cjk": int(cjk.sum()),
        "incomplete_summaries": int(incomplete.sum()),
        "invalid_sentiments": int(invalid_sentiment.sum()),
    }


def apply_hardcoded_manual_genai_repairs(df, df_name="dataframe"):
    """
    Fill the 12 remaining missing GenAI summaries manually, as a post-processing repair step.
    """
    manual_repairs = {
        "Sertraline": (
            "Reduced anxiety, intrusive thoughts, and headaches with manageable side effects.",
            "positive",
        ),
        "Ethinyl estradiol / levonorgestrel": (
            "Improved periods but caused dryness, low libido, and acne.",
            "negative",
        ),
        "Chateal": (
            "Caused weight gain, mood changes, nausea, insomnia, and pain.",
            "negative",
        ),
        "Mononessa": (
            "Cleared acne but caused weight gain, low libido, and pain.",
            "negative",
        ),
        "Tioconazole": (
            "Made itching and discomfort much worse.",
            "negative",
        ),
        "Venlafaxine": (
            "Reduced appetite while improving energy, clarity, and confidence.",
            "positive",
        ),
        "Budesonide": (
            "Stopped diarrhea and restored daily functioning.",
            "positive",
        ),
        "Neupro": (
            "Patch greatly restored ability to complete daily tasks.",
            "positive",
        ),
        "Ethinyl estradiol / norgestimate": (
            "Improved skin and appearance despite dehydration and prolonged bleeding.",
            "positive",
        ),
        "Lurasidone": (
            "Brought lasting calm despite increased hunger.",
            "positive",
        ),
        "Blisovi 24 Fe": (
            "Regulated periods but caused severe nausea, infections, and cramps.",
            "negative",
        ),
        "Carmol 20": (
            "Kept skin soft and smooth for years.",
            "positive",
        ),
    }

    possible_drug_cols = ["drug name", "drugName", "drug"]
    possible_summary_cols = ["generated summary", "generated_summary"]
    possible_sentiment_cols = ["identified sentiment", "identified_sentiment"]

    drug_col = next((c for c in possible_drug_cols if c in df.columns), None)
    summary_col = next((c for c in possible_summary_cols if c in df.columns), None)
    sentiment_col = next((c for c in possible_sentiment_cols if c in df.columns), None)

    if drug_col is None or summary_col is None or sentiment_col is None:
        print(f"[{df_name}] Skipping manual repairs: required columns were not found.")
        return df

    # Only repair rows where the summary is currently missing or empty
    missing_summary_mask = (
        df[summary_col].isna()
        | (df[summary_col].astype(str).str.strip() == "")
        | (df[summary_col].astype(str).str.lower().str.strip() == "nan")
    )

    applied = 0

    for drug_name, (summary, sentiment) in manual_repairs.items():
        mask = (
            missing_summary_mask
            & (df[drug_col].astype(str).str.strip() == drug_name)
        )

        n_matches = int(mask.sum())

        if n_matches > 0:
            df.loc[mask, summary_col] = summary
            df.loc[mask, sentiment_col] = sentiment
            applied += n_matches

    print(f"[{df_name}] Applied hardcoded manual GenAI repairs: {applied}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="final_genai_outputs",
        help="Directory containing the GenAI output CSV files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    paths = {
        "alt1_final": output_dir / "alternative_1_roberta_metadata_final.csv",
        "alt2_final": output_dir / "alternative_2_bert_base_final.csv",
        "alt1_progress": output_dir / "alternative_1_roberta_metadata_progress_with_debug.csv",
        "alt2_progress": output_dir / "alternative_2_bert_base_progress_with_debug.csv",
    }

    print("Loading files...")
    dfs = {name: load_csv(path) for name, path in paths.items()}

    print("Building automatic repairs from alternative 1 progress/debug file...")
    repairs = build_repairs(dfs["alt1_progress"])

    report_dir = output_dir / "postprocess_repair"
    report_dir.mkdir(parents=True, exist_ok=True)

    repairs_path = report_dir / "genai_rows_repair_report.csv"
    repairs.to_csv(repairs_path, index=False)

    unresolved = (
        repairs[repairs["repair_status"] == "manual_review_required"]
        if not repairs.empty
        else pd.DataFrame()
    )

    unresolved_path = report_dir / "genai_rows_needing_manual_review_before_hardcoded_repairs.csv"
    unresolved.to_csv(unresolved_path, index=False)

    print(f"Repair candidates found: {len(repairs)}")
    print(f"Manual review required before hardcoded repairs: {len(unresolved)}")
    print(f"Repair report saved to: {repairs_path}")

    print("Applying automatic resolved repairs to all final/progress files...")
    fixed_dfs = {name: apply_repairs(df, repairs) for name, df in dfs.items()}

    print("Applying hardcoded manual repairs for the 12 reviewed missing summaries...")
    fixed_dfs = {
        name: apply_hardcoded_manual_genai_repairs(df, name)
        for name, df in fixed_dfs.items()
    }

    fixed_paths = {
        name: path.with_name(path.stem + "_repaired.csv")
        for name, path in paths.items()
    }

    print("Saving repaired files...")
    for name, df in fixed_dfs.items():
        df.to_csv(fixed_paths[name], index=False)
        print(f"Saved: {fixed_paths[name]}")

    print("\nQuality diagnostics after automatic + hardcoded repair:")
    diagnostics = pd.DataFrame([
        summarize_quality(df, name) for name, df in fixed_dfs.items()
    ])

    diagnostics_path = report_dir / "genai_repaired_quality_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    print(diagnostics.to_string(index=False))
    print(f"Diagnostics saved to: {diagnostics_path}")

    issue_columns = [
        "missing_summary",
        "summaries_over_10_words",
        "summaries_with_cjk",
        "incomplete_summaries",
        "invalid_sentiments",
    ]

    remaining_issues = diagnostics[issue_columns].sum().sum()

    if remaining_issues > 0:
        print("\nSome issues still remain after hardcoded repairs.")
        print("Check the diagnostics file above before using the outputs for submission.")
    else:
        print("\nAll detected GenAI output issues were repaired successfully.")
        print("The *_repaired.csv files are now the clean final CSV files.")


if __name__ == "__main__":
    main()