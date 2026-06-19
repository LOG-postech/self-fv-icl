# Datasets

Running

```bash
python prepare_data.py        # from the repository root
```

reconstructs every dataset into `dataset_files/icl/` (git-ignored) in the common schema
consumed by `main_icl/utils/data_utils.py` (loaded as
`dataset_files/icl/<data>_{train,test}.json`). Build a subset with
`python prepare_data.py --only wordnet moons`.

NLTK WordNet data is needed once: `python -c "import nltk; nltk.download('wordnet')"`.

## What gets built, and from where

| Dataset | `--data` names | Source | License |
|---|---|---|---|
| WordNetMCQ (single / multi / triple / quad) + OOD variants | `wordnet`, `wordnet_multi`, `wordnet_triple`, `wordnet_quad`, `wordnet_test_ood1`–`ood5` | constructed from **NLTK WordNet** | WordNet 3.0 License (Princeton) |
| Two-moons | `moons` | `sklearn.datasets.make_moons` (synthetic) | — |
| AG News | `ag_news` | HF `ag_news` | per dataset card |
| Emotion | `emotion` | HF `SetFit/emotion` (Saravia et al. 2018 split) | per dataset card |
| GSM8K | `gsm8k` | HF `gsm8k` (config `main`) | MIT |
| HellaSwag | `hellaswag` | HF `Rowan/hellaswag` | MIT |

The WordNet-derived and `moons` datasets are the paper's own construction; the remaining
tasks are downloaded from the Hugging Face Hub at build time and reformatted.

## Schema

Each record is a dict with `input`, `output`, `label`, `choices`, `choices_label`, `id`
(WordNet multi/triple/quad use list-valued `output`/`label`/`choices_label`; HellaSwag adds a
`context` field). See `prepare_data.py` for the exact construction of each.
