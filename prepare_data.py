#!/usr/bin/env python3
"""
Regenerate the ICL task datasets into ``dataset_files/icl/``.

No task data is distributed with this repository; this script reconstructs every
dataset from its original source so that the experiments in ``main_icl/`` can run.

  - WordNetMCQ (wordnet / wordnet_multi / wordnet_triple / wordnet_quad) and the OOD
    query variants are built from NLTK WordNet (the paper's own construction).
  - moons is generated synthetically with scikit-learn.
  - ag_news, emotion, gsm8k and hellaswag are downloaded from the Hugging Face Hub and
    reformatted into the common schema.

Run from the repository root:

    python prepare_data.py                 # build everything
    python prepare_data.py --only wordnet moons
    python prepare_data.py --only ag_news emotion gsm8k hellaswag

NLTK WordNet data is required once:  python -c "import nltk; nltk.download('wordnet')"
"""

import argparse
import collections
import json
import os
import random

OUT_DIR = "dataset_files/icl"
CHOICE_LABELS = ["A", "B", "C", "D"]
WORDNET_LEVELS = [4, 5, 6, 7]


def _save(data, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path} ({len(data)} examples)")


def _split_and_save(dataset, base_name, seed=42):
    """Shuffle, 80/20 split, assign sequential ids, and save train/test."""
    rng = random.Random(seed)
    rng.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train, test = dataset[:split_idx], dataset[split_idx:]
    for i, ex in enumerate(train):
        ex["id"] = i
    for i, ex in enumerate(test):
        ex["id"] = len(train) + i
    _save(train, f"{base_name}_train")
    _save(test, f"{base_name}_test")


# --------------------------------------------------------------------------- #
# WordNet
# --------------------------------------------------------------------------- #
def _clean(synset_name):
    """'residual_oil.n.01' -> 'residual oil'."""
    return synset_name.split(".")[0].replace("_", " ")


def _target_dict(target_level):
    """Map each parent synset at ``target_level`` to its (sorted) child synset names."""
    from nltk.corpus import wordnet as wn

    target, queue, visited = {}, collections.deque([(wn.synset("physical_entity.n.01"), 0)]), set()
    while queue:
        cur, depth = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        if depth == target_level:
            children = sorted(c.name() for c in cur.hyponyms())
            if children:
                target[cur.name()] = children
        if depth < target_level + 1:
            for hyp in cur.hyponyms():
                if hyp not in visited:
                    queue.append((hyp, depth + 1))
    return target


def build_wordnet_single(seed=42):
    """WordNetMCQ1: single-answer 4-way multiple choice."""
    random.seed(seed)
    dataset = []
    for level in WORDNET_LEVELS:
        td = _target_dict(level)
        pairs = [(p, c[0]) for p, c in td.items() if len(c) == 1]
        all_children = [c for children in td.values() for c in children]
        for parent, child in pairs:
            others = [c for c in all_children if c != child]
            wrong = random.sample(others, min(3, len(others)))
            choices = [_clean(child)] + [_clean(w) for w in wrong]
            random.shuffle(choices)
            idx = choices.index(_clean(child))
            dataset.append({
                "input": f"Which of the following is/are types of {_clean(parent)}?",
                "output": _clean(child),
                "label": idx,
                "choices": {CHOICE_LABELS[i]: c for i, c in enumerate(choices)},
                "choices_label": CHOICE_LABELS[idx],
                "level": f"{level}-{level + 1}",
                "parent_synset": parent,
                "child_synset": child,
            })
    _split_and_save(dataset, "wordnet", seed)


def build_wordnet_nanswer(n_correct, base_name, seed=42):
    """WordNetMCQ2-style: ``n_correct`` correct answers + (4 - n_correct) distractors."""
    random.seed(seed)
    num_wrong = 4 - n_correct
    dataset = []
    for level in WORDNET_LEVELS:
        td = _target_dict(level)
        groups = [(p, c) for p, c in td.items() if len(c) == n_correct]
        all_children = [c for children in td.values() for c in children]
        for parent, children in groups:
            children_clean = [_clean(c) for c in children]
            if len(set(children_clean)) != len(children_clean):
                continue
            others = [c for c in all_children if c not in children]
            wrong = random.sample(others, min(num_wrong, len(others)))
            choices = children_clean + [_clean(w) for w in wrong]
            if len(choices) != 4:
                continue
            random.shuffle(choices)
            correct_idx = [choices.index(c) for c in children_clean]
            dataset.append({
                "input": f"Which of the following is/are types of {_clean(parent)}?",
                "output": children_clean,
                "label": correct_idx,
                "choices": {CHOICE_LABELS[i]: c for i, c in enumerate(choices)},
                "choices_label": sorted(CHOICE_LABELS[i] for i in correct_idx),
                "level": f"{level}-{level + 1}",
                "parent_synset": parent,
                "child_synsets": children,
            })
    _split_and_save(dataset, base_name, seed)


def build_wordnet_ood():
    """OOD-query variants (ood1-5) derived from the generated wordnet train/test files."""
    import generate_ood_wordnet
    generate_ood_wordnet.main()


# --------------------------------------------------------------------------- #
# moons (synthetic)
# --------------------------------------------------------------------------- #
def build_moons(seed=42):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_moons

    x, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
    df = pd.DataFrame({"x1": np.round(x[:, 0], 2), "x2": np.round(x[:, 1], 2), "label": y})
    df = df.rename_axis("index").reset_index()
    df["input_text"] = df.apply(lambda r: f"x={r['x1']:.2f}, y={r['x2']:.2f}", axis=1)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_size = int(len(df) * 0.2)
    name = {0: "A", 1: "B"}
    choices = {"A": "A", "B": "B"}

    def fmt(rows, id_offset=0):
        out = []
        for _, r in rows.iterrows():
            out.append({
                "input": r["input_text"],
                "output": name[r["label"]],
                "label": int(r["label"]),
                "choices": choices,
                "choices_label": name[r["label"]],
                "id": int(r["index"]) + id_offset,
            })
        return out

    test = fmt(df[:test_size])
    train = fmt(df[test_size:])
    end = max(item["id"] for item in train) + 1
    test = fmt(df[:test_size], id_offset=end)
    _save(train, "moons_train")
    _save(test, "moons_test")


# --------------------------------------------------------------------------- #
# Hugging Face datasets
# --------------------------------------------------------------------------- #
def _classification_records(rows, label_to_name, choices, choices_label, id_offset=0):
    out = []
    for idx, row in enumerate(rows):
        name = label_to_name[row["label"]]
        out.append({
            "input": row["text"],
            "output": name,
            "label": row["label"],
            "choices": choices,
            "choices_label": choices_label[name],
            "id": idx + id_offset,
        })
    return out


def build_emotion():
    from datasets import load_dataset

    # SetFit/emotion mirrors the standard Saravia et al. (2018) 16k/2k split with stable
    # ordering (the default dair-ai/emotion config now serves the much larger "unsplit" set).
    ds = load_dataset("SetFit/emotion")
    names = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
    choices = {"A": "Sadness", "B": "Joy", "C": "Love", "D": "Anger", "E": "Fear", "F": "Surprise"}
    choices_label = {v: k for k, v in choices.items()}
    train = _classification_records(ds["train"], names, choices, choices_label)
    test = _classification_records(ds["test"], names, choices, choices_label, id_offset=len(train))
    _save(train, "emotion_train")
    _save(test, "emotion_test")


def build_ag_news():
    from datasets import load_dataset

    ds = load_dataset("ag_news")["test"].train_test_split(test_size=0.1, shuffle=False, seed=42)
    names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    choices = {"A": "World", "B": "Sports", "C": "Business", "D": "Sci/Tech"}
    choices_label = {v: k for k, v in choices.items()}
    train = _classification_records(ds["train"], names, choices, choices_label)
    test = _classification_records(ds["test"], names, choices, choices_label, id_offset=len(train))
    _save(train, "ag_news_train")
    _save(test, "ag_news_test")


def build_gsm8k(seed=42):
    from datasets import load_dataset

    random.seed(seed)
    ds = load_dataset("gsm8k", "main")

    def fmt(rows, id_offset=0):
        out = []
        for idx, row in enumerate(rows):
            choices = {k: random.randint(-100, 100) for k in CHOICE_LABELS}
            label = random.choice(CHOICE_LABELS)
            answer = row["answer"].strip().split("####")[-1].strip()
            choices[label] = int(answer.replace(",", ""))
            out.append({
                "input": row["question"].strip(),
                "output": answer,
                "choices": choices,
                "choices_label": label,
                "label": ord(label) - ord("A"),
                "id": idx + id_offset,
            })
        return out

    train = fmt(ds["train"])
    test = fmt(ds["test"], id_offset=len(train))
    _save(train, "gsm8k_train")
    _save(test, "gsm8k_test")


def build_hellaswag(n_train=8000, n_test=2000):
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag")
    question = "Which of the following is the best ending to the given context?"

    def fmt(rows, id_offset=0):
        out = []
        for idx, row in enumerate(rows):
            label = int(row["label"])
            endings = list(row["endings"])
            out.append({
                "context": row["ctx"],
                "input": question,
                "output": CHOICE_LABELS[label],
                "choices": {CHOICE_LABELS[i]: e for i, e in enumerate(endings)},
                "choices_label": CHOICE_LABELS[label],
                "label": label,
                "id": idx + id_offset,
            })
        return out

    # The HF `test` split is unlabeled, so we draw the labeled test set from `validation`.
    train = fmt(ds["train"].select(range(min(n_train, len(ds["train"])))))
    test = fmt(ds["validation"].select(range(min(n_test, len(ds["validation"])))), id_offset=len(train))
    _save(train, "hellaswag_train")
    _save(test, "hellaswag_test")


BUILDERS = {
    "wordnet": lambda: (build_wordnet_single(), build_wordnet_ood()),
    "wordnet_multi": lambda: build_wordnet_nanswer(2, "wordnet_multi"),
    "wordnet_triple": lambda: build_wordnet_nanswer(3, "wordnet_triple"),
    "wordnet_quad": lambda: build_wordnet_nanswer(4, "wordnet_quad"),
    "moons": build_moons,
    "ag_news": build_ag_news,
    "emotion": build_emotion,
    "gsm8k": build_gsm8k,
    "hellaswag": build_hellaswag,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(BUILDERS), default=list(BUILDERS),
                        help="Subset of datasets to build (default: all).")
    args = parser.parse_args()
    for name in args.only:
        print(f"[{name}]")
        BUILDERS[name]()
    print("Done.")


if __name__ == "__main__":
    main()
