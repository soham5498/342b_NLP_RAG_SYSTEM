
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Datasets
ds_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
ds_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
ds_unlabeled = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
dataset2_knowledge = load_dataset("medalpaca/medical_meadow_wikidoc")
ds_query = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

# Initialize Sentence Transformer Model for Embedding on GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Extract and Prepare Knowledge Base
def extract_knowledge(datasets):
    """
    Extract contexts from multiple datasets and prepare the knowledge base.
    """
    knowledge_base = []
    
    # Extract from ds_artificial, ds_labeled, ds_unlabeled
    for dataset in datasets[:-1]:
        for row in dataset["train"]["context"]:
            if isinstance(row, dict) and "contexts" in row:
                knowledge_base.extend(row["contexts"])
    
    # Extract from dataset2_knowledge
    knowledge_base.extend(datasets[-1]["train"]["output"])
    
    return list(set(knowledge_base))  # Deduplicate entries

# Combine all knowledge
knowledge_base = extract_knowledge([ds_artificial, ds_labeled, ds_unlabeled, dataset2_knowledge])
print(f"Total unique contexts in knowledge base: {len(knowledge_base)}")

# Embed the Knowledge Base on GPU
print("Generating embeddings for the knowledge base...")
knowledge_embeddings = model.encode(knowledge_base, convert_to_tensor=True, device=device)
print("Knowledge base embeddings generated.")

def retrieve_contexts(query_text, knowledge_base, knowledge_embeddings, top_k=3, max_tokens=700):
    """
    Retrieve the most relevant contexts for a given query using embeddings.
    """
    # Encode query on GPU
    query_embedding = model.encode(query_text, convert_to_tensor=True, device=device)
    
    # Compute similarity (remains on GPU)
    similarities = util.cos_sim(query_embedding, knowledge_embeddings)[0]
    
    # Move similarities tensor to CPU before converting to NumPy
    similarities_cpu = similarities.cpu().numpy()
    
    # Sort by descending similarity
    top_indices = np.argsort(-similarities_cpu)[:top_k]
    
    # Combine top contexts up to 700 tokens
    retrieved_contexts = []
    token_count = 0
    for idx in top_indices:
        context = knowledge_base[idx]
        token_count += len(context.split())  # Approximate token count
        if token_count > max_tokens:
            break
        retrieved_contexts.append(context)
    
    # Return combined contexts (just a substring if you need strict 700-char limit)
    return " ".join(retrieved_contexts)

# Limit ds_query to the first 20 questions
ds_query_limited = ds_query.select(range(20))  

# Process the queries
results = []
for i, row in enumerate(ds_query_limited):
    question = row["question"]
    options = row["options"]
    
    result = {"question": question, "options": {}}
    
    # Retrieve contexts for each option
    for key, option in options.items():
        context = retrieve_contexts(option, knowledge_base, knowledge_embeddings)
        result["options"][key] = {
            "option_text": option, 
            "retrieved_context": context
        }
    
    results.append(result)

# Display Results for Questions
for res in results:  
    print(f"\nQuestion: {res['question']}")
    for key, val in res["options"].items():
        print(f"Option {key}: {val['option_text']}")
        # Print a truncated snippet of the context
        print(f"Context: {val['retrieved_context'][:700]}...")  

