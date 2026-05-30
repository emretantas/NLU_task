---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
- natural-language-inference
- nli
repo: https://github.com/VP1606/nlu-cw

---

# Model Card for VP1606-CS101-o-emretantas-NLI

<!-- Provide a quick summary of what the model is/does. -->

OracleNet is a binary Natural Language Inference (NLI) classifier that determines
      whether a hypothesis is entailed by a given premise. It is built on a custom ResESIM architecture
      and takes as input rich pre-computed token embeddings combining GloVe, ELMo, CharCNN, POS tag,
      and negation features.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

OracleNet (run ff2f02d4) consists of two components:
      (1) A ResESIM encoder: a highway input projection maps 1477-d token embeddings to a 512-d hidden
      space, followed by 3 stacked ESIM blocks each performing 8-head cross-attention between premise
      and hypothesis, with residual connections.
      (2) A StockClassifier: independent aggregation BiLSTMs for premise and hypothesis, followed by
      attention-weighted pooling, masked max pooling, and masked mean pooling. The resulting vectors
      are concatenated (including difference and element-wise product terms) and passed through a
      2-layer FFN to produce class logits.

- **Developed by:** Premakantha Varun, Kaan Oktem and Munir Emre Tantas
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** ResESIM (Residual Enhanced Sequential Inference Model)
- **Finetuned from model [optional]:** N/A — trained from scratch on pre-computed embeddings (GloVe 300d + ELMo 1024d + CharCNN 100d + POS 50d + Negation 3d = 1477d)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/VP1606/nlu-cw
- **Paper or documentation:** Li et al. (2019) "Residual Connected Enhanced Sequential Inference Model for Natural Language Inference." IEEE Access.

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,432 NLI premise-hypothesis pairs (COMP34812 NLI track training set).
      Input features are pre-computed and concatenated per token:
      GloVe (300d) + ELMo (1024d) + CharCNN (100d) + POS tags (50d) + Negation (3d) = 1477d.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - input_dim: 1477
      - hidden_dim: 512
      - num_blocks: 3
      - num_attn_heads: 8
      - dropout_rate: 0.2
      - learning_rate: 1e-04
      - batch_size: 32
      - num_epochs: 15 (best at epoch 8)
      - warmup_steps: 200 (~5% of total steps)
      - total_steps: 11,445
      - optimizer: Adam (β1=0.9, β2=0.999, weight_decay=0.01)
      - loss: CrossEntropyLoss

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - model size: ~83 MB (best_model.pt)
      - best checkpoint: epoch 8 / 15
      - total training steps: 11,445

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

COMP34812 NLI track development set.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Macro F1-score
      - Accuracy

### Results

Run ff2f02d4 (best checkpoint, epoch 8):
      - Dev Macro F1:  70.41%
      - Dev Accuracy:  70.52%
      - Train Macro F1: 72.60%
      - Train Accuracy: 72.76%

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 500 MB (model + embeddings)
      - GPU: optional — supports Apple MPS and CUDA; falls back to CPU

### Software


      - Python 3.10
      - PyTorch 2.10.0
      - NumPy 1.26.4
      - spaCy 3.3.3
      - scikit-learn 1.7.2
      - Optuna 4.2.1 (hyperparameter search)

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model relies on GloVe and ELMo embeddings, which may encode
      social biases present in their pre-training corpora (Wikipedia, Common Crawl, 1 Billion Word
      Benchmark). The model is trained for binary NLI (entailment vs. non-entailment) and is not
      designed for 3-class NLI tasks. Performance may degrade on out-of-domain text.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters were determined via automated search using Optuna.

      References for techniques used in this model:

      Base architecture:
      - Li et al. (2019) "Residual Connected Enhanced Sequential Inference Model for Natural
        Language Inference." IEEE Access.

      Intra-sentence self-attention (applied within each ESIM block before cross-attention):
      - Parikh et al. (2016) "A Decomposable Attention Model for Natural Language Inference."
        EMNLP 2016.
      - Wang & Jiang (2017) "A Compare-Aggregate Model for Matching Text Sequences." ICLR 2017.

      Attention-weighted pooling (used in StockClassifier alongside max and mean pooling):
      - Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning to Align and
        Translate." ICLR 2015.
      - dos Santos et al. (2016) "Attentive Pooling Networks." arXiv:1602.03609.

      Highway input projection (maps heterogeneous 1477-d input to 512-d hidden space):
      - Srivastava et al. (2015) "Training Very Deep Networks." NeurIPS 2015.
