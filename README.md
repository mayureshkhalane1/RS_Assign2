# SASRec Sequential Recommendation 

This project implements a **sequential recommendation model** based on SASRec using PyTorch.  
The goal is to predict the next item a user will interact with based on their historical behavior sequence.

---

## Project Structure

```
.
├── ratings.dat # MovieLens 1M dataset (required)
├── dataset.py # Data preprocessing + dataset construction
├── model.py # SASRec model implementation
├── train.py # Training loop + optimization
├── eval.py # Evaluation metrics and ranking logic
├── main.py # Experiment orchestration
└── README.md
```
---

## Dataset

The dataset file `ratings.dat` must be placed in the root directory.

Format:
```
userId::movieId::rating::timestamp
```

### Preprocessing Steps

1. **Implicit feedback conversion**
   - Keep interactions where `rating >= 4`
   - Discard all other interactions

2. **Chronological ordering**
   - Sort each user’s interactions by timestamp

3. **User filtering**
   - Remove users with fewer than 5 interactions

4. **ID remapping**
   - Map user IDs and item IDs to contiguous integers
   - `0` is reserved for padding

5. **Leave-one-out split**
   For each user sequence:

[i1, i2, i3, ..., i(n-2), i(n-1), i(n)]

Train: [i1, i2, ..., i(n-2)]

Valid: i(n-1)

Test: i(n)


---

## Processed Data Representation

### Training Data (token-level)

Each user sequence is converted into multiple training signals:

Example:

User sequence: [10, 23, 45, 67]

Input: [0, 10, 23, 45]

Pos: [0, 23, 45, 67]

Neg: [0, n1, n2, n3] (random negatives)

```
- Input = sequence prefix
- Pos = next item
- Neg = randomly sampled item not in user history
```


---

### Evaluation Data

For each user:

- Validation:
```
Input: train sequence
Target: validation item
```

- Test:
```
Input: train + validation
Target: test item
```

---

## Model Architecture (SASRec)

The model follows a Transformer-style architecture:

### 1. Embedding Layer
- Item embeddings: map item IDs → vectors

- Positional embeddings: encode sequence order


x = item_embedding + positional_embedding


---

### 2. Self-Attention Blocks

Each block contains:
```
- Multi-head self-attention
- Causal mask (prevents future leakage)
- Residual connection
- Layer normalization
- Feedforward network
- Dropout
```
---

### 3. Causal Mask

Ensures:

position t can only attend to positions ≤ t


---

### 4. Sequence Representation

- Use the **last position hidden state**
- Represents the entire sequence

---

### 5. Prediction

Score items using dot product:


score = sequence_representation · item_embedding


---

## Training

### Objective

Binary Cross Entropy with negative sampling:
```
- Positive: actual next item
- Negative: randomly sampled item
```
---

### Optimization
```
- Optimizer: Adam
- Early stopping based on validation NDCG@10
- Gradient clipping applied
```
---

## Evaluation Metrics

The model is evaluated using ranking metrics:
```
- Recall@10, Recall@20  
- NDCG@10, NDCG@20  
```
### Evaluation Modes

- `sampled`: rank target among sampled negatives (fast)

- `full`: rank target among all unseen items (strict, slower)

---

## Running the Project

### Basic Run

```bash
python main.py
```

All experiments are defined in main.py:
```
experiment_configs = {
    "config_small": { ... },
    "config_deeper": { ... },
}
```

### Configurable Parameters

Model Architecture
```
hidden_dim — embedding size
num_blocks — number of attention layers
num_heads — number of attention heads
max_seq_len — sequence length
dropout — dropout rate
```
Training
```
lr — learning rate
batch_size
epochs
early_stop_patience
grad_clip
```
Evaluation
```
eval_mode — "sampled" or "full"
eval_num_negatives — number of negatives (sampled mode)
```
Example Custom Config
```
"my_config": {
    **base_config,
    "hidden_dim": 128,
    "num_blocks": 4,
    "num_heads": 4,
    "max_seq_len": 100,
    "dropout": 0.2,
    "lr": 1e-3,
}
```
### Important Constraints
hidden_dim must be divisible by num_heads

