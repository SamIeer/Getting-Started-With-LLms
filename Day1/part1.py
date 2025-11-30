import numpy as np

# Step 1 - Tokenization 
# "GETTING STARTED WITH LLMS" -----> Tokenization ------> ["GETTING", "STARTED", "WITH", "LLMS"]
def tokenize(text):
    tokens = text.lower().split()
    return tokens

sentence = "Transformers are amazing for language modeling"
print(tokenize(sentence))

# Step 2 Build Vocubulary 
def build_vocab(tokens_list):
    vocab = {}
    for tokens in tokens_list:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

sentences = [
    tokenize("I am learning about large language models"),
    tokenize("Language models learn from text"),
    tokenize("Transfomers improve language understanding")
]

vocab = build_vocab(sentences)
print(vocab)

# Step 3 Convert Tokens  ---> Integers
def tokens_to_ids(tokens, vocab):
    return [vocab[t] for t in tokens]

toks = tokenize("language models learn")
ids = tokens_to_ids(toks, vocab)
print(ids)

# Step 4 Word Embeddings 
# We need to convert token ids into vectors that represent meaning
# Create Embedding Matrix
vocab_size = len(vocab)
embedding_dim = 8 

# Embedding matrix of shape (vocab size X embedding dim)
embeddings = np.random.randn(vocab_size, embedding_dim)

print("Embedding matrix shape:", embeddings.shape)
# 13 tokens in vocab and Each word is represented as a vector of length 8

# Step 5 Converting tokens IDs --> embeddings 
def embed(ids):
    return embeddings[ids]

ids = [5,6,7] # "language models learn"
vecs = embed(ids)
print(vecs)
# Each row = vector meaning of a word
# embeddings[word_id] → vector
# "cat" → [0.13, -0.55, 0.92, ...]
# "dog" → [0.11, -0.52, 0.89, ...]
# cat ≈ dog
# cat ≠ democracy
# This is how meaning is encoded numerically.

# IMPORTANT CONCEPT
# The model will gradually adjust these vectors during training until similar words have similar vectors.
# For example:
# king - man + woman ≈ queen
# That happens because embeddings capture semantic relationships.

# Step 5 Self Attention 
# Self-attention allows each word to look at other words in the sentence and decide which are important.
# Code --- Compute Q,K, V vectors
def self_attention(x):
    d = x.shape[1] # embedding dim

    # Weight matrices 
    Wq = np.random.randn(d,d)
    Wk = np.random.randn(d,d)
    Wv = np.random.randn(d,d)

    # compute queries, keys, values
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # attention weights 
    scores = Q @ K.T # similarity scores
    weights = softmax(scores)

    # combine values
    output = weights @ V

    return output, weights

def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp/ np.sum(exp, axis=-1, keepdims=True)

#Run self-attention on your embedding vecotrs
out, weights = self_attention(vecs)
print("Attention weights:\n", weights)
print("Output vectors:\n",out)

# 🧠 What is happening here?

# Let’s break this down in human terms:
# Embeddings (you already have)
# Each word → vector
# Q (Query)
# What am I looking for?
# K (Key)
# What does this word contain?
# V (Value)
# What information does this word pass along?
out, weights = self_attention(vecs)
print(weights)
print(out)

# After that

# We will build:

# ✔ a Transformer block
# ✔ residual connections
# ✔ layer normalization
# ✔ feed-forward network

# And then…
# 🚀 we’ll TRAIN IT to predict next tokens.