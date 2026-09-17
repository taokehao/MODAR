# Official Multisource Data

The dataset is already downloaded, enriched, built, and merged. The canonical
description is:

```text
data_prep/official_multisource/DATASET_OVERVIEW.md
```

Default model input:

```text
data_prep/outputs/official_unified_multimodal/{train,valid,test}.lmdb
```

Final split sizes:

```text
train  10,003,856
valid     101,978
test      101,853
```

Daily commands:

```bash
python data_prep/official_multisource/scripts/manage_dataset.py status

python data_prep/official_multisource/scripts/manage_dataset.py inspect \
  --split train --n 3

python data_prep/official_multisource/scripts/manage_dataset.py export-jsonl
```

The numbered build/download scripts under `scripts/` are retained for provenance
and reproducibility. They should not be rerun unless intentionally regenerating
the dataset.
