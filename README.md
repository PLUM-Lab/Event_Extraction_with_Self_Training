# Learning from a Friend: Improving Event Extraction via Self-Training with Feedback from Abstract Meaning Representation

Official implementation of the Findings of ACL 2023 paper:

> Zhiyang Xu, Jay-Yoon Lee, and Lifu Huang. 2023. [Learning from a Friend: Improving Event Extraction via Self-Training with Feedback from Abstract Meaning Representation](https://aclanthology.org/2023.findings-acl.662/). Findings of ACL 2023.

## Method

Self-Training with Feedback (STF) improves event extraction by using Abstract Meaning Representation (AMR) to assess pseudo-labeled event arguments. The method has three main components:

1. An event extraction model predicts event structures for unlabeled sentences.
2. An AMR-path scoring model measures the compatibility between each predicted event argument and its corresponding AMR path.
3. The compatibility scores weight the pseudo-label training loss, providing structured feedback during self-training.

STF uses AMR to provide feedback during training and does not add an AMR requirement at inference time. Inference inputs otherwise follow the selected event extraction backbone.

The implementation supports the two event extraction backbones and three datasets evaluated in the paper:

| Backbone | Datasets |
| --- | --- |
| OneIE | ACE05-E, ACE05-E+, ERE-EN |
| AMR-IE | ACE05-E, ACE05-E+, ERE-EN |

## Setup

The experiments use Python 3.8, PyTorch 1.10.2, CUDA 11.3, DGL 0.7.2, Transformers 4.14.1, and RoBERTa-large. Create the provided Conda environment and add the repository root to `PYTHONPATH`:

```bash
conda env create -f old_code/env.yml
conda activate stf
export PYTHONPATH="$(pwd)"
```

The experiments in the paper were run on one NVIDIA A100 GPU.

AMR parsing uses the [transition-based AMR parser](https://github.com/IBM/transition-amr-parser). Follow its installation instructions and make its `transition_amr_parser` package available before preparing AMR files:

```bash
export PYTHONPATH="YOUR_AMR_PARSER_PATH:${PYTHONPATH}"
```

### Configure paths

Replace the following placeholders with paths on your machine:

- `YOUR_PROJECT_PATH`: absolute path to this repository.
- `YOUR_ACE05_SOURCE_PATH`: raw ACE 2005 data directory.
- `YOUR_ERE_SOURCE_PATH`: raw ERE data directory.
- `YOUR_BERT_CACHE_PATH`: Hugging Face model cache directory.
- `YOUR_AMR_PARSER_PATH`: transition-based AMR parser installation or checkout.
- `YOUR_AMR_PARSER_CHECKPOINT_PATH`: transition-based AMR parser checkpoint.
- `YOUR_IE_MODEL_PATH`: trained event extraction checkpoint.
- `YOUR_SCORE_MODEL_PATH`: trained AMR-path scoring checkpoint.

JSON configuration files require literal paths. To find placeholders that still need to be replaced, run:

```bash
rg -n 'YOUR_[A-Z0-9_]*PATH' \
  config/one_stage_*.json \
  config/seq_score/*.json \
  script/process_amr.py
```

## Data preparation

The paper evaluates event extraction on:

- ACE 2005 (LDC2006T06)
- ERE-EN

It uses two unlabeled corpora for STF:

- AMR 3.0 (LDC2020T02)
- New York Times Annotated Corpus (LDC2008T19), with system-generated AMR graphs

After preparation, arrange the data as follows:

```text
dataset/
├── ACE05-E/roberta_large/
│   ├── train.oneie.json
│   ├── dev.oneie.json
│   ├── test.oneie.json
│   ├── train_amrs.pkl
│   ├── dev_amrs.pkl
│   └── test_amrs.pkl
├── ACE05+/roberta_large/...
├── ere/roberta_large/...
├── aligned_amr3.0.amr-ie/
│   └── aux_amr3.0.jsonl
└── aligned_amr3.0/
    ├── order_list_ace.json
    ├── order_list_ace+.json
    └── order_list_ere.json
```

The `.oneie.json` files contain event extraction annotations. The `*_amrs.pkl` files contain AMR parser outputs aligned with the labeled sentences and are used to train the compatibility scorer and AMR-IE. `aux_amr3.0.jsonl` contains the AMR 3.0 sentences and graphs used for STF.

### Prepare ACE05-E and ACE05-E+

For ACE05-E:

```bash
python preprocessing/process_ace.py \
  -i YOUR_ACE05_SOURCE_PATH \
  -o YOUR_PROJECT_PATH/dataset/ACE05-E/roberta_large \
  -s resource/splits/ACE05-E \
  -b roberta-large \
  -c YOUR_BERT_CACHE_PATH \
  -l english
```

For ACE05-E+, use an `ACE05+` output directory and add `--time_and_val`:

```bash
python preprocessing/process_ace.py \
  -i YOUR_ACE05_SOURCE_PATH \
  -o YOUR_PROJECT_PATH/dataset/ACE05+/roberta_large \
  -s resource/splits/ACE05-E \
  -b roberta-large \
  -c YOUR_BERT_CACHE_PATH \
  -l english \
  --time_and_val
```

### Prepare ERE-EN

```bash
python preprocessing/process_ere.py \
  -i YOUR_ERE_SOURCE_PATH \
  -o YOUR_PROJECT_PATH/dataset/ere/roberta_large \
  -b roberta-large \
  -c YOUR_BERT_CACHE_PATH \
  -l english \
  -d normal
```

### Parse the labeled data with AMR

Set `YOUR_AMR_PARSER_CHECKPOINT_PATH` in `script/process_amr.py`. Update the train, development, and test paths in its main block for the target dataset, then run:

```bash
python script/process_amr.py
```

This produces the `train_amrs.pkl`, `dev_amrs.pkl`, and `test_amrs.pkl` files used by AMR-IE and the compatibility scorer.

### Prepare AMR 3.0 auxiliary data

The paper uses JAMR-aligned AMR 3.0 graphs. Prepare an aligned AMR file with `# ::id`, `# ::tok`, `# ::node`, `# ::root`, and `# ::edge` metadata, update the input and output paths in `amr_processing/process_gold_amr.py`, and run:

```bash
python amr_processing/process_gold_amr.py
python script/convert2roberta.py \
  -i YOUR_PROJECT_PATH/dataset/aligned_amr3.0.amr-ie/training_all.txt \
  -o YOUR_PROJECT_PATH/dataset/aligned_amr3.0.amr-ie/aux_amr3.0.jsonl
```

Each STF configuration also takes an `order_file`. This is a JSON list ordered by the desired example priority:

```json
[
  ["sentence-id-1", 0.91],
  ["sentence-id-2", 0.87]
]
```

The first 75% is used for self-training and the remaining 25% is used for compatibility-based model selection.

For the system-AMR experiments in the paper, parse the sampled NYT 2004 sentences with the transition-based AMR parser, convert them to the same auxiliary-data format, and set `aux_train_file` in the corresponding `config/system_amr/*.json` configuration.

## Training

The reproduction workflow below has three steps. The commands use ACE05-E and GPU 0.

Use the corresponding configurations for another dataset:

| Dataset | Scoring model | OneIE + STF | AMR-IE + STF |
| --- | --- | --- | --- |
| ACE05-E | `config/seq_score/roberta_ace.json` | `config/one_stage_baseline.json` | `config/one_stage_amr-ie_ace.json` |
| ACE05-E+ | `config/seq_score/roberta_ace+.json` | `config/one_stage_baseline_ace+.json` | `config/one_stage_amr-ie_ace+.json` |
| ERE-EN | `config/seq_score/roberta_ere.json` | `config/one_stage_baseline_ere.json` | `config/one_stage_amr-ie_ere.json` |

### 1. Train the event extraction model used by the scorer

```bash
mkdir -p experiment/ie/ACE
python train.py \
  -c config/roberta_oneie-e.json \
  -n ie/ACE/roberta_exp2 \
  -g 0 \
  -ep 80
```

The checkpoint is saved to:

```text
experiment/ie/ACE/roberta_exp2/best.role.mdl
```

For ACE05-E+ or ERE-EN, use `config/roberta_oneie+.json` or `config/roberta_oneie-ere.json`.

### 2. Train the AMR-path compatibility scorer

Set `ie_model_path` in the selected `config/seq_score/*.json` file to the checkpoint from the previous stage. Then run:

```bash
mkdir -p experiment/seq_score/ACE
python score_train.py \
  -c config/seq_score/roberta_ace.json \
  -n seq_score/ACE/roberta_exp2 \
  -g 0 \
  -i 2 \
  -ie oneie
```

The compatibility-scoring checkpoint is saved to:

```text
experiment/seq_score/ACE/roberta_exp2/best.role.mdl
```

### 3. Train an event extraction model with STF

Set `score_model_path`, `aux_train_file`, and `order_file` in the selected `config/one_stage_*.json` file.

To train OneIE with STF:

```bash
mkdir -p experiment/stf/ACE
python one_stage_self_train.py \
  -c config/one_stage_baseline.json \
  -n stf/ACE/oneie_seed42 \
  -g 0 \
  -s 42 \
  -f \
  -ie oneie \
  -i 2 \
  -lr 0.001 \
  -blr 0.00001 \
  -m 1.0 \
  -a 10 \
  -st 0.9 \
  -pe 10
```

To train AMR-IE with STF, also set `train_amr`, `dev_amr`, and `test_amr` in the configuration:

```bash
mkdir -p experiment/stf/ACE
python one_stage_self_train.py \
  -c config/one_stage_amr-ie_ace.json \
  -n stf/ACE/amrie_seed42 \
  -g 0 \
  -s 42 \
  -f \
  -ie amrie \
  -i 2 \
  -lr 0.001 \
  -blr 0.00001 \
  -m 1.0 \
  -a 10 \
  -st 0.9 \
  -pe 10
```

Each STF run saves the selected checkpoint, training log, and development/test predictions under `experiment/<run-name>/`.

## Evaluation

STF training reports event extraction metrics on the development and test sets after each epoch. The selected checkpoint and predictions are saved as:

```text
experiment/<run-name>/best.role.mdl
experiment/<run-name>/result.dev.json
experiment/<run-name>/result.test.json
```

To evaluate a OneIE checkpoint separately, use a configuration matching its dataset:

```bash
mkdir -p experiment/eval
python self_eval.py \
  -c config/one_stage_baseline.json \
  -n eval/oneie_ace \
  -g 0 \
  -m YOUR_IE_MODEL_PATH
```

The metrics are written to `experiment/eval/oneie_ace/eval_results.json`.

To evaluate the AMR-path compatibility scorer:

```bash
python score_eval.py \
  -c config/seq_score/roberta_ace.json \
  -g 0 \
  -m YOUR_IE_MODEL_PATH \
  -s YOUR_SCORE_MODEL_PATH
```

## Acknowledgements

This implementation builds on the following research projects:

- **OneIE v0.4.8** by Ying Lin, Heng Ji, Fei Huang, and Lingfei Wu: [software](https://blender.cs.illinois.edu/software/oneie/) and [paper](https://aclanthology.org/2020.acl-main.713/). The base event extraction model, graph representation, global features, preprocessing, and training infrastructure are based on OneIE.
- **AMR-IE** by Zixuan Zhang and Heng Ji: [code](https://github.com/zhangzx-uiuc/AMR-IE) and [paper](https://aclanthology.org/2021.naacl-main.4/). The AMR graph encoder and AMR-aware event extraction components are adapted from AMR-IE.
- **Transition-based AMR Parser**: [code](https://github.com/IBM/transition-amr-parser) and [paper](https://aclanthology.org/2020.findings-emnlp.89/). It is used to generate aligned AMR graphs for the event extraction datasets.
- **JAMR** by Jeffrey Flanigan and collaborators: [code](https://github.com/jflanigan/jamr). JAMR alignments are used to prepare the AMR 3.0 auxiliary data.

Please cite the relevant upstream work when using these components.

## Citation

```bibtex
@inproceedings{xu-etal-2023-learning,
  title = {Learning from a Friend: Improving Event Extraction via Self-Training with Feedback from Abstract Meaning Representation},
  author = {Xu, Zhiyang and Lee, Jay-Yoon and Huang, Lifu},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year = {2023},
  pages = {10421--10437},
  url = {https://aclanthology.org/2023.findings-acl.662/},
  doi = {10.18653/v1/2023.findings-acl.662}
}
```
