from input_output_pair import GPTDataset,create_dataloader_v1
import re
import torch
import torch.nn as nn
import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")

vocab_size = tokenizer.n_vocab       # 50257
embedding_dim = 256

#opening the file
file = "/data/vansh/LLM_FROM_SCRATCH/archive/the-verdict.txt"

with open(file, "r", encoding="utf-8") as f:
    raw_text = f.read()


embeding_dim=256

embeding_matrix=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embeding_dim)



dataloader=create_dataloader_v1(raw_text1=raw_text)
for input_tokens, target_tokens in dataloader:

    input_embeddings = embeding_matrix(input_tokens)

    print("Input IDs shape:", input_tokens.shape)
    print("Embedding shape:", input_embeddings.shape)
    print("Target IDs shape:", target_tokens.shape)

    break
