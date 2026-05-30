# OracleNet — COMP34812 NLU Coursework
**Group:** CW47
**Category:** B — Deep Learning (non-transformer)
**Model:** ResESIM (Residual ESIM) with 1477-dimensional input embeddings

---

## Results

| Split | Macro F1 |
|---|---|
| Dev set | 70.41% |
| Training set | 72.6% |
| Trial set (50 examples) | 75.85% |

**Best model:** `ff2f02d4`

---

## Code Structure

```
NLU-CW/
├── demo.ipynb                  # Demo notebook — run this to generate predictions
├── res_esim/
│   ├── run/
│   │   └── train.py            # Training entry point
│   ├── model_layers/
│   │   └── oracle_net.py       # OracleNet model definition (ResESIM)
│   ├── loader/
│   │   └── res_esim_dataset.py # Dataset loader for pre-computed embeddings
│   └── evaluation/             # Evaluation code (dev set F1)
├── precomputeClasses.py        # EmbeddingPrecomputer — 1477d embedding pipeline
├── build_meta.py               # Builds meta.pt (vocab, GloVe, char2idx, pos2idx)
├── final_model_versions/
│   └── ff2f02d4/
│       ├── best_model.pt       # Trained model weights (stored via Git LFS)
│       └── meta.json           # Model configuration
├── notebook_data/
│   ├── meta.pt                 # Vocab + GloVe embeddings (stored via Git LFS)
│   └── elmo_model/             # ELMo weights (stored via Git LFS)
├── data/
│   ├── train.csv               # Training data
│   ├── dev.csv                 # Development data (with labels)
│   ├── test.csv                # Test data (no labels)
│   └── NLI_trial.csv           # Trial data (50 examples, with labels)
└── require310.txt              # Python dependencies
```

**Code organisation:**
- **Training code:** `res_esim/run/train.py` and `precomputeClasses.py`
- **Evaluation code:** `res_esim/evaluation/` — evaluates on dev set
- **Demo code:** `demo.ipynb` — loads saved model and generates predictions

---

## Running the Demo

### Requirements
- Python 3.10
- Git LFS (to pull model weights)

### Setup
```bash
# Pull model weights via Git LFS
git lfs install
git lfs pull

# Install dependencies
pip install -r require310.txt
./venv310/bin/pip install 'tokenizers==0.13.3' 'allennlp==2.10.1' 'allennlp-models==2.10.1' 'numpy<2.0'

```

### Generate Predictions
Open and run `demo.ipynb` top to bottom. It will:
1. Pre-compute 1477d embeddings (GloVe + ELMo + CharCNN + POS + Negation)
2. Load the trained OracleNet model
3. Run inference on `data/test.csv`
4. Save predictions to `Group_16_B.csv`

---

## Training

```bash
# From the root directory
python -m res_esim.run.train
```

Note: ELMo pre-computation requires the `venv310` environment (Python 3.10 with AllenNLP).

---

## Attribution

This work builds on the following:

| Resource | Reference |
|---|---|
| ESIM architecture | Chen et al. (2017), "Enhanced LSTM for Natural Language Inference" |
| ELMo embeddings | Peters et al. (2018), "Deep contextualized word representations" — weights from [AllenNLP](https://allennlp.org/) |
| GloVe embeddings | Pennington et al. (2014), "GloVe: Global Vectors for Word Representation" — 300d vectors from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) |
| CharCNN | Kim et al. (2016), "Character-Aware Neural Language Models" |
| POS tags | Universal Dependencies via [spaCy](https://spacy.io/) `en_core_web_sm` |
| AllenNLP library | Gardner et al. (2018), [https://github.com/allenai/allennlp](https://github.com/allenai/allennlp) |

---

## Model Files (Cloud Storage)

Large files are stored via Git LFS. If Git LFS is unavailable, download manually:

| File | Description |
|---|---|
| `notebook_data/meta.pt` | Vocab + GloVe embedding matrix (1.0 MB) |
| `final_model_versions/ff2f02d4/best_model.pt` | Trained OracleNet weights (86.9 MB) |
| `notebook_data/elmo_model/weights.hdf5` | ELMo weights (357 MB) |