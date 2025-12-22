"""
Be4 training:
dim_768_cosine_ndcg@10: 0.10107309358309094
dim_512_cosine_ndcg@10: 0.09863679086449742
dim_256_cosine_ndcg@10: 0.08915869908768258
dim_128_cosine_ndcg@10: 0.0801744438174278
dim_64_cosine_ndcg@10: 0.0563580388865545

After training:
dim_768_cosine_ndcg@10: 0.09166409042111621
dim_512_cosine_ndcg@10: 0.09089315889482967
dim_256_cosine_ndcg@10: 0.08660039122171372
dim_128_cosine_ndcg@10: 0.08250279026712513
dim_64_cosine_ndcg@10: 0.07268121239909381
"""

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets

model_id = "models/bge-base-en-v1.5_mrl_20251216_173407"  # Hugging Face model ID
matryoshka_dimensions = [768, 512, 256, 128, 64] # Important: large to small
 
# Load a model
model = SentenceTransformer(
    model_id, device="cuda" if torch.cuda.is_available() else "cpu"
)
 
# load test dataset
test_dataset = load_dataset("json", data_files="dataset/test_dataset.json", split="train")
train_dataset = load_dataset("json", data_files="dataset/train_dataset.json", split="train")
corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
 
# Convert the datasets to dictionaries
corpus = dict(
    zip(corpus_dataset["id"], corpus_dataset["positive"])
)  # Our corpus (cid => document)
queries = dict(
    zip(test_dataset["id"], test_dataset["anchor"])
)  # Our queries (qid => question)
 
# Create a mapping of relevant document (1 in our case) for each query
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for q_id in queries:
    relevant_docs[q_id] = [q_id]
 
 
matryoshka_evaluators = []
# Iterate over the different dimensions
for dim in matryoshka_dimensions:
    print(f"Evaluating at dimension {dim}...")
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"dim_{dim}",
        truncate_dim=dim,  # Truncate the embeddings to a certain dimension
        score_functions={"cosine": cos_sim},
    )
    matryoshka_evaluators.append(ir_evaluator)
 
# Create a sequential evaluator
evaluator = SequentialEvaluator(matryoshka_evaluators)

# Evaluate the model
results = evaluator(model)
 
# # COMMENT IN for full results
# print(results)
 
# Print the main score
for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print
    print(f"{key}: {results[key]}")