# compute_negation.py
import pandas as pd
import torch

TRIGGERS = {
    "not",
    "no",
    "never",
    "n't",
    "neither",
    "nor",
    "without",
    "lack",
    "lacking",
    "fail",
    "fails",
    "failed",
    "hardly",
    "barely",
    "scarcely",
}
BOUNDARIES = {",", ".", "but", "and", "or", "though", "although"}


def negation_flags(text: str) -> list[int]:
    tokens = text.lower().split()  # simple whitespace tokenisation
    flags, in_scope = [], False
    for tok in tokens:
        if tok in TRIGGERS:
            in_scope = True
        if tok in BOUNDARIES:
            in_scope = False
        flags.append(1 if in_scope else 0)
    return flags


for split in ["train", "dev"]:
    df = pd.read_csv(f"data/{split}.csv")  # columns: premise, hypothesis, entailment
    out = {
        "premise_negation": [negation_flags(p) for p in df["premise"]],
        "hypothesis_negation": [negation_flags(h) for h in df["hypothesis"]],
    }
    torch.save(out, f"output/{split}_negation.pt")
    print(f"{split}: {len(df)} examples saved")
