# Datasets

## Demonstration samples (`data/demo/demonstration_samples.json`)

Manually curated messages for examiner laboratory tests:

- Valid phishing / legitimate examples
- Insufficient-evidence example
- Contradictory-evidence example
- Fabricated-evidence base text + invented phrase

**License:** Original academic content (project authors). Free for educational use with this prototype.

## Evaluation / training sets (`data/evaluation/`)

- `train_dataset.json` — synthetic messages to fit the local TF-IDF classifier
- `eval_dataset.json` — labelled evaluation samples with annotated evidence spans

**Source:** Authored for this project (not scraped from live mailboxes).

**Distinction:** These are **demonstration / evaluation** samples, not real-world production validation data.

## External public datasets (optional future work)

Public phishing corpora (e.g., IWSPA-AP, Nazario, or similar academic collections) could be used for larger experiments. This prototype ships with synthetic data so the project runs fully offline without redistributing third-party corpora. If you add an external dataset, document its source, license, and train/eval split in this file.

## Disclaimer

Scores on these small synthetic sets demonstrate the evaluation pipeline. They do **not** claim production phishing-detection performance.
