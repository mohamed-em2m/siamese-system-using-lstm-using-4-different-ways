# Question Similarity Neural Network Models

This repository contains a PyTorch implementation of different neural network architectures for identifying duplicate questions in a dataset. The models use word embeddings and bidirectional LSTMs to encode questions and compare their semantic similarity.

## Overview

The notebook trains four different models for the task of question pair classification:

1. **Basic LSTM Model**: A bidirectional LSTM with absolute difference similarity measurement
2. **Cosine Similarity Model**: A model that uses cosine similarity to determine question pair similarity
3. **Contrastive Loss Model**: Implements a contrastive loss function for similarity learning
4. **Categorical Model**: A model that outputs classification probabilities using cross-entropy loss

## Dataset

The code uses the "Question Pairs Dataset" which contains pairs of questions labeled as duplicate or not duplicate. Each question is preprocessed by:
- Tokenizing using `simple_preprocess`
- Converting tokens to indices using a built vocabulary
- Padding sequences to a fixed length (212 tokens)

## Model Architectures

### Common Components

All models share similar components:
- Word embedding layer (100 dimensions)
- Bidirectional LSTM encoders
- Concat operation to combine forward and backward LSTM states

### Model Variants

1. **Basic Model (sts)**
   - Uses absolute difference between question vectors
   - Loss: BCE with Logits Loss

2. **Cosine Similarity Model (sts_cosin)**
   - Uses cosine similarity between encoded questions
   - Loss: Cosine Embedding Loss

3. **Contrastive Model (sts_contrastive)**
   - Uses euclidean distance with contrastive loss function
   - Loss: Custom contrastive loss

4. **Categorical Model (sts_cat)**
   - Element-wise multiplication of encoded questions
   - Linear layer to output class probabilities
   - Loss: Cross-Entropy Loss

## Training Process

The notebook includes:
- Model definitions
- Training loops for each model variant
- Evaluation on validation sets
- Learning rate scheduling with ReduceLROnPlateau
- Accuracy calculation

## Performance

The categorical model (sts_cat) achieved the best performance, reaching:
- 100% accuracy on the training set
- ~63% accuracy on the validation set

The models were saved to disk for later use.

## Inference Examples

The notebook includes example inference for different question pairs:
- "What is the meaning of life" vs "How old is life"
- "How old are you" vs "What is your age"
- "Why do humans die" vs "Why when I wake up I think I am dead"

Each model outputs similarity scores or classifications for these examples.

## Requirements

- PyTorch
- torchtext
- gensim
- numpy
- pandas
- fasttext
- NLTK
- Matplotlib

## Usage

The trained models can be used to determine if two questions are semantically similar:

```python
def predict_similarity(q1, q2, model_type='categorical'):
    # Preprocess questions
    q1_tensor = change(q1).cuda()
    q2_tensor = change(q2).cuda()
    
    # Select model and get prediction
    if model_type == 'categorical':
        output = model4(q1_tensor, q2_tensor)
        return torch.argmax(output).item()
    elif model_type == 'basic':
        output = model(q1_tensor, q2_tensor)
        return nn.Sigmoid()(output).item()
    # Add other model types as needed
```
