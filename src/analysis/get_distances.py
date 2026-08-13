"""
Proof-of-concept pipeline step: build the R-ready trial-level CSV that adds
LCSH-based structural variables (same_heading, hier_distance, same_nbhd) and
LCSH-frame BERT distances (distance_bert_lcsh) to Trott & Bergen's Experiment 2
trials, restricted to the 25 stimulus words in our stratified sample.

ELMo (distance_elmo_lcsh) is left blank: the original ELMo stack (allennlp +
its pinned torch/mxnet deps) does not install under this project's Python
version, and there is no drop-in modern replacement. The column is still
emitted (all-NaN) so downstream R code can detect and handle its absence.

Inputs:
    data/processed/selected_headings.csv  (sense_uid, selected_heading_uri, authorized_label)
    data/raw/stimuli.csv                  (Word, M1_a, M1_b, M2_a, M2_b, ...)
    data/processed/sense_hier_metrics.csv (sense_uid, hier_distance, ...)
    data/raw/polysemy_s2_final.csv        (Trott & Bergen Experiment 2 trial data)

Output:
    data/processed/s2_trials_lcsh_distances.csv
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

### PATHS
SELECTED_HEADINGS_PATH = Path("data/processed/selected_headings.csv")
STIMULI_PATH = Path("data/raw/stimuli.csv")
SENSE_HIER_METRICS_PATH = Path("data/processed/sense_hier_metrics.csv")
TRIALS_PATH = Path("data/raw/polysemy_s2_final.csv")
SAVE_PATH = Path("data/processed/s2_trials_lcsh_distances.csv")

### CONSTANTS
NBHD_THRESHOLD = 3  # "D" in the Cognitive Warrant theory
SAME_NBHD_MAX_DISTANCE = 2 * NBHD_THRESHOLD  # hier_distance <= this => same_nbhd
BERT_MODEL_NAME = "bert-base-uncased"

SENSE_UID_RE = re.compile(r"^(.+)_M(\d+)$")


# ---------------------------------------------------------------------
# Step 1-2: build the sense-level lookup (mapped_word, sentence_a/b,
# heading_uri, heading_label) and filter trials to our 25 stimulus words
# ---------------------------------------------------------------------


def build_sense_lookup(
    selected_headings_path: Path, stimuli_path: Path
) -> pd.DataFrame:
    selected = pd.read_csv(selected_headings_path)
    if not {"sense_uid", "selected_heading_uri", "authorized_label"} <= set(
        selected.columns
    ):
        raise ValueError(
            "selected_headings.csv must contain sense_uid, selected_heading_uri, authorized_label"
        )

    stimuli = pd.read_csv(stimuli_path)
    stim_by_word = {
        str(r["Word"]).strip(): r for _, r in stimuli.iterrows() if pd.notna(r["Word"])
    }

    rows = []
    for _, r in selected.iterrows():
        sense_uid = str(r["sense_uid"]).strip()
        m = SENSE_UID_RE.match(sense_uid)
        if not m:
            raise ValueError(f"sense_uid {sense_uid!r} does not match '<word>_M<n>'")
        mapped_word, m_num = m.group(1), m.group(2)

        stim_row = stim_by_word.get(mapped_word)
        if stim_row is None:
            raise ValueError(f"No stimuli.csv row found for word {mapped_word!r}")

        sentence_a = stim_row[f"M{m_num}_a"]
        sentence_b = stim_row[f"M{m_num}_b"]

        rows.append(
            {
                "sense_uid": sense_uid,
                "mapped_word": mapped_word,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "selected_heading_uri": str(r["selected_heading_uri"]).strip(),
                "authorized_label": r["authorized_label"],
            }
        )

    return pd.DataFrame(rows)


def load_hier_distance_map(sense_hier_metrics_path: Path) -> Dict[str, float]:
    sense_hier = pd.read_csv(sense_hier_metrics_path)
    return dict(zip(sense_hier["sense_uid"], sense_hier["hier_distance"]))


def filter_trials(trials_path: Path, mapped_words: set) -> pd.DataFrame:
    trials = pd.read_csv(trials_path)
    return trials[trials["word"].isin(mapped_words)].reset_index(drop=True).copy()


# ---------------------------------------------------------------------
# Step 3: parse version_with_order into prime/target sense_uid + sentence key
# ---------------------------------------------------------------------


def parse_prime_target(word: str, version_with_order: str) -> Tuple[str, str, str, str]:
    """
    version_with_order e.g. "M1_a_M2_a" -> prime portion "M1_a", target portion "M2_a".
    Returns (sense_uid_prime, sentence_key_prime, sense_uid_target, sentence_key_target).
    """
    parts = str(version_with_order).split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected version_with_order format: {version_with_order!r}")
    prime_m, prime_key, target_m, target_key = parts
    sense_uid_prime = f"{word}_{prime_m}"
    sense_uid_target = f"{word}_{target_m}"
    return sense_uid_prime, prime_key, sense_uid_target, target_key


# ---------------------------------------------------------------------
# Step 4: BERT embeddings under the two alternative frames
# ---------------------------------------------------------------------


class BertSpanEmbedder:
    def __init__(self, model_name: str = BERT_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed_span(self, sentence: str, span_text: str) -> np.ndarray:
        """Mean-pooled last-hidden-state embedding of `span_text` as it occurs in `sentence`."""
        char_start = sentence.find(span_text)
        if char_start == -1:
            raise ValueError(f"Span {span_text!r} not found in sentence {sentence!r}")
        char_end = char_start + len(span_text)

        enc = self.tokenizer(
            sentence, return_offsets_mapping=True, return_tensors="pt"
        )
        offsets = enc.pop("offset_mapping")[0].tolist()

        token_idx = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (s == e == 0) and s < char_end and e > char_start
        ]
        if not token_idx:
            raise ValueError(
                f"Could not map span {span_text!r} to any tokens in {sentence!r}"
            )

        out = self.model(**enc)
        hidden = out.last_hidden_state[0]  # (seq_len, hidden_dim)
        span_vec = hidden[token_idx].mean(dim=0)
        return span_vec.numpy()


def frame1_sentence(label: str) -> str:
    return f"The topic of this resource is {label}."


def frame2_sentence(label: str, sentence: str) -> str:
    return f"Regarding {label}: {sentence}"


def precompute_frame_embeddings(
    sense_lookup: pd.DataFrame, embedder: BertSpanEmbedder
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """
    Frame 1 depends only on the heading label -> keyed by label.
    Frame 2 depends on (sense_uid, sentence_key 'a'/'b') -> keyed by that pair.
    Both are small, fixed sets, so we precompute once instead of per trial.
    """
    frame1_cache: Dict[str, np.ndarray] = {}
    frame2_cache: Dict[Tuple[str, str], np.ndarray] = {}

    labels = sense_lookup["authorized_label"].unique().tolist()
    for label in tqdm(labels, desc="BERT frame-1 (same_heading=0) embeddings"):
        frame1_cache[label] = embedder.embed_span(frame1_sentence(label), label)

    for _, r in tqdm(
        list(sense_lookup.iterrows()), desc="BERT frame-2 (same_heading=1) embeddings"
    ):
        sense_uid = r["sense_uid"]
        label = r["authorized_label"]
        for key in ("a", "b"):
            sentence = r[f"sentence_{key}"]
            frame2_cache[(sense_uid, key)] = embedder.embed_span(
                frame2_sentence(label, sentence), label
            )

    return frame1_cache, frame2_cache


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine(a, b))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    sense_lookup = build_sense_lookup(SELECTED_HEADINGS_PATH, STIMULI_PATH)
    sense_info: Dict[str, dict] = {
        r["sense_uid"]: r.to_dict() for _, r in sense_lookup.iterrows()
    }
    mapped_words = set(sense_lookup["mapped_word"])
    hier_distance_map = load_hier_distance_map(SENSE_HIER_METRICS_PATH)

    trials = filter_trials(TRIALS_PATH, mapped_words)
    print(f"Filtered to {len(trials)} trials across {len(mapped_words)} stimulus words")

    print(f"Loading BERT model ({BERT_MODEL_NAME})...")
    embedder = BertSpanEmbedder(BERT_MODEL_NAME)
    frame1_cache, frame2_cache = precompute_frame_embeddings(sense_lookup, embedder)

    out_rows = []
    skipped = 0
    for _, trial in tqdm(trials.iterrows(), total=len(trials), desc="Building trial records"):
        word = str(trial["word"]).strip()
        try:
            sense_uid_prime, key_prime, sense_uid_target, key_target = parse_prime_target(
                word, trial["version_with_order"]
            )
            prime = sense_info[sense_uid_prime]
            target = sense_info[sense_uid_target]
        except (ValueError, KeyError):
            skipped += 1
            continue

        sentence_prime = prime[f"sentence_{key_prime}"]
        sentence_target = target[f"sentence_{key_target}"]
        heading_uri_prime = prime["selected_heading_uri"]
        heading_uri_target = target["selected_heading_uri"]
        heading_label_prime = prime["authorized_label"]
        heading_label_target = target["authorized_label"]

        same_heading = heading_uri_prime == heading_uri_target
        if same_heading:
            hier_distance = 0.0
            same_nbhd = 1
            d_bert = cosine_distance(
                frame2_cache[(sense_uid_prime, key_prime)],
                frame2_cache[(sense_uid_target, key_target)],
            )
        else:
            hier_distance = hier_distance_map.get(sense_uid_prime, np.inf)
            same_nbhd = int(hier_distance <= SAME_NBHD_MAX_DISTANCE)
            d_bert = cosine_distance(
                frame1_cache[heading_label_prime], frame1_cache[heading_label_target]
            )

        out_rows.append(
            {
                "word": word,
                "version_with_order": trial["version_with_order"],
                "sense_uid_prime": sense_uid_prime,
                "sentence_prime": sentence_prime,
                "heading_uri_prime": heading_uri_prime,
                "heading_label_prime": heading_label_prime,
                "sense_uid_target": sense_uid_target,
                "sentence_target": sentence_target,
                "heading_uri_target": heading_uri_target,
                "heading_label_target": heading_label_target,
                "same_heading": same_heading,
                "hier_distance": hier_distance,
                "same_nbhd": same_nbhd,
                "distance_bert_lcsh": d_bert,
                "distance_elmo_lcsh": np.nan,  # see module docstring: ELMo unavailable
                # passthrough from Trott & Bergen's original Experiment 2 processing,
                # for comparison against the LCSH-frame distances above
                "distance_bert": trial["distance_bert"],
                "distance_elmo": trial["distance_elmo"],
                "rt": trial["rt"],
                "correct_response": trial["correct_response"],
                "log_rt": trial["log_rt"],
                "prior_rt": trial["prior_rt"],
                "ambiguity_type": trial["ambiguity_type"],
                "Class": trial["Class"],
                "same": trial["same"],
                # passthrough identifiers useful for QA / joins back to the raw trial
                "subject": trial["subject"],
                "trial_index": trial["trial_index"],
                "item": trial["item"],
            }
        )

    if skipped:
        print(f"Skipped {skipped} trials due to unparseable version_with_order or missing sense lookup")

    out_df = pd.DataFrame(out_rows)

    required_nonnull = ["distance_bert_lcsh", "log_rt", "prior_rt", "ambiguity_type"]
    for col in required_nonnull:
        n_null = out_df[col].isna().sum()
        if n_null:
            print(f"WARNING: {n_null} rows have null {col}")

    out_df.to_csv(SAVE_PATH, index=False)
    print(f"\nWrote {len(out_df)} rows -> {SAVE_PATH}")
    print(
        "NOTE: distance_elmo_lcsh is entirely NaN in this output "
        "(ELMo/allennlp is not available in this environment)."
    )


if __name__ == "__main__":
    main()
