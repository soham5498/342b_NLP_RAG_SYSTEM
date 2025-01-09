# Refrences
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options?row=9
# https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc
# https://huggingface.co/datasets/qiaojin/PubMedQA
# https://chatgpt.com/ - for code structuring and the docstring
# https://huggingface.co/docs/transformers/en/model_doc/bart
# https://pytorch.org/
# https://github.com/facebookresearch/faiss


import os
import pickle
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import faiss
from transformers import BartTokenizer, BartForConditionalGeneration
import random


class TextProcessor:
    """
    Handles text chunking and processing.

    Attributes
    ----------
    chunk_size : int
        The size of each text chunk.
    overlap : int
        The overlap size between consecutive chunks.
    """
    def __init__(self, chunk_size=300, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """
        Splits text into fixed-length chunks with overlap.

        Parameters
        ----------
        text : str
            The input text to be chunked.

        Returns
        -------
        list of str
            A list of text chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks


class DataProcessor:
    """
            Processes datasets into chunked data.

            Attributes
            ----------
            text_processor : TextProcessor
                An instance of TextProcessor for chunking text.
    """
    def __init__(self, text_processor):
        self.text_processor = text_processor

    def process_row(self, row, is_labeled):
        """
        Processes a single dataset row.

        Parameters
        ----------
        row : dict
            A single row from the dataset.
        is_labeled : bool
            Whether the row contains labeled data.

        Returns
        -------
        list of dict
            A list of dictionaries containing chunked data.
        """
        raw_text = row.get("context", {}).get("contexts", []) if is_labeled else row.get("output", "")
        raw_text = " ".join(raw_text) if isinstance(raw_text, list) else raw_text
        chunks = self.text_processor.chunk_text(raw_text)

        chunked_data = []
        for chunk in chunks:
            chunk_data = {"chunk_text": chunk}
            if is_labeled:
                chunk_data.update({
                    "label": row.get("context", {}).get("labels", ["unlabeled"])[0],
                    "meshes": row.get("context", {}).get("meshes", ["no_mesh"])
                })
            chunked_data.append(chunk_data)

        return chunked_data

    def process_dataset(self, dataset, is_labeled=False):
        """
        Processes an entire dataset.

        Parameters
        ----------
        dataset : list of dict
            The dataset to process.
        is_labeled : bool, optional
            Whether the dataset contains labeled data (default is False).

        Returns
        -------
        list of dict
            A list of dictionaries containing chunked data for the entire dataset.
        """
        chunked_dataset = []
        for row in tqdm(dataset, desc="Processing Dataset"):
            chunked_rows = self.process_row(row, is_labeled)
            chunked_dataset.extend(chunked_rows)
        return chunked_dataset


class EmbeddingManager:
    """
    Handles embedding generation and management.

    Attributes
    ----------
    model : SentenceTransformer
        The SentenceTransformer model for generating embeddings.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def generate_embeddings(self, data):
        """
        Generates embeddings for the given data.

        Parameters
        ----------
        data : list of dict
            The data for which to generate embeddings.

        Returns
        -------
        np.ndarray
            An array of embeddings.
        """
        texts = [chunk["chunk_text"] for chunk in data]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    @staticmethod
    def save_embeddings(data, file_path):
        """Saves embeddings to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_embeddings(file_path):
        """Loads embeddings from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


class MetadataManager:
    """Handles metadata generation and management."""
    @staticmethod
    def generate_metadata(chunked_data):
        """
        Generates metadata from chunked data.

        Parameters
        ----------
        chunked_data : list of dict
            The chunked data for which to generate metadata.

        Returns
        -------
        list of dict
            A list of metadata dictionaries.
        """
        metadata = []
        for chunk in chunked_data:
            metadata_entry = {
                "chunk_text": chunk["chunk_text"],
                "label": chunk.get("label", "unlabeled"),
                "meshes": chunk.get("meshes", []),
                "embedding": chunk.get("embedding", None)
            }
            metadata.append(metadata_entry)
        return metadata

    @staticmethod
    def save_metadata(metadata, file_path):
        """Saves metadata to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load_metadata(file_path):
        """Loads metadata from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


class FAISSManager:
    """
    Manages the FAISS index for efficient vector similarity search.

    Attributes
    ----------
    dimension : int
        Dimensionality of the embeddings.
    index : faiss.IndexFlatL2
        The FAISS index for similarity search.
    """
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        """
        Adds embeddings to the FAISS index.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to add to the FAISS index.
        """
        self.index.add(np.array(embeddings, dtype=np.float32))

    def save_index(self, file_path):
        """Saves the FAISS index to a file."""
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path):
        """Loads the FAISS index from a file."""
        self.index = faiss.read_index(file_path)

    def search(self, query_embedding, top_k):
        """
        Searches the FAISS index for the most similar embeddings.

        Parameters
        ----------
        query_embedding : np.ndarray
            The query embedding to search for.
        top_k : int
            The number of top results to return.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Distances and indices of the top-k closest embeddings.
        """
        return self.index.search(query_embedding.reshape(1, -1), top_k)


class QueryHandler:
    """
    Handles query and retrieval tasks.

    Attributes
    ----------
    embedding_manager : EmbeddingManager
        The manager for generating and managing embeddings.
    faiss_manager : FAISSManager
        The manager for the FAISS index.
    metadata : list of dict
        The metadata associated with the indexed embeddings.
    """
    def __init__(self, embedding_manager, faiss_manager, metadata):
        self.embedding_manager = embedding_manager
        self.faiss_manager = faiss_manager
        self.metadata = metadata

    def query_options_enhanced_with_keywords(self, question, options, top_k=5, question_keywords=None, option_keywords=None):
        """
        Queries the embeddings for each option and retrieves relevant contexts, with optional keyword prioritization.

        Parameters
        ----------
        question : str
            The question text.
        options : list of str
            The list of answer options.
        top_k : int, optional
            The number of top results to retrieve (default is 5).
        question_keywords : list of str, optional
            Keywords to emphasize in the question (default is None).
        option_keywords : list of str, optional
            Keywords to emphasize in the options (default is None).

        Returns
        -------
        dict
            A dictionary where keys are options and values are retrieved contexts.
        """
        results = {}
        all_retrieved_chunks = set()  # Track globally retrieved chunks for uniqueness

        for option in options:
            # Combine option and question with keyword emphasis
            query_text = f"Option: {option}. Question: {question}. Focus on relevant keywords."
            query_embedding = self.embedding_manager.model.encode([query_text], convert_to_numpy=True)

            # Add keyword embeddings to query embedding
            if question_keywords or option_keywords:
                combined_keywords = (question_keywords or []) + (option_keywords or [])
                keyword_embeddings = self.embedding_manager.model.encode(combined_keywords, convert_to_numpy=True)
                keyword_embedding = np.mean(keyword_embeddings, axis=0)  # Aggregate keyword embeddings
                query_embedding = 0.7 * query_embedding + 0.3 * keyword_embedding  # Adjust weights as needed

            # Ensure query embedding is in the correct format
            query_embedding = query_embedding.astype("float32")

            # Search in the FAISS index
            distances, indices = self.faiss_manager.search(query_embedding.reshape(1, -1), top_k)

            # Retrieve chunks and prioritize uniqueness
            retrieved_chunks = [
                {
                    "chunk_text": self.metadata[idx]["chunk_text"],
                    "label": self.metadata[idx].get("label", "unlabeled"),
                    "distance": distances[0][i]
                }
                for i, idx in enumerate(indices[0])
                if self.metadata[idx]["chunk_text"] not in all_retrieved_chunks
            ]

            # Add retrieved chunks to global set
            all_retrieved_chunks.update(chunk["chunk_text"] for chunk in retrieved_chunks)

            # Post-retrieval keyword filtering and scoring
            if question_keywords or option_keywords:
                combined_keywords = (question_keywords or []) + (option_keywords or [])
                retrieved_chunks = sorted(
                    retrieved_chunks,
                    key=lambda chunk: sum(
                        1 for keyword in combined_keywords if keyword.lower() in chunk["chunk_text"].lower()
                    ),
                    reverse=True  # Prioritize chunks with more keyword matches
                )

            # Store results
            results[option] = {"retrieved_chunks": retrieved_chunks}

        return results


class Summarizer:
    """
    Handles text summarization using a BART model.

    Attributes
    ----------
    tokenizer : BartTokenizer
        The tokenizer for the BART model.
    model : BartForConditionalGeneration
        The BART model for summarization.
    """
    def __init__(self, model_name="facebook/bart-base"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=100, min_length=30):
        """
        Summarizes the input text.

        Parameters
        ----------
        text : str
            The text to summarize.
        max_length : int, optional
            The maximum length of the summary (default is 100).
        min_length : int, optional
            The minimum length of the summary (default is 30).

        Returns
        -------
        str
            The generated summary.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Example pipeline setup
if __name__ == "__main__":
    text_processor = TextProcessor()
    data_processor = DataProcessor(text_processor)
    embedding_manager = EmbeddingManager()
    faiss_manager = FAISSManager()
    summarizer = Summarizer()

    # Load datasets
    print("Loading datasets...")
    context_ds_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial")['train']
    context_ds_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled")['train']
    context_ds_unlabeled = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")['train']
    context_ds_knowledge = load_dataset("medalpaca/medical_meadow_wikidoc")['train']
    query_dataset = load_dataset("GBaker/MedQA-USMLE-4-options")['train']

    # Check for existing embeddings, metadata, and FAISS index
    embeddings_file = "embeddings.pkl"
    metadata_file = "metadata.pkl"
    faiss_index_file = "faiss_index.bin"

    if os.path.exists(embeddings_file) and os.path.exists(metadata_file) and os.path.exists(faiss_index_file):
        print("Loading existing embeddings, metadata, and FAISS index...")
        embeddings = EmbeddingManager.load_embeddings(embeddings_file)
        metadata = MetadataManager.load_metadata(metadata_file)
        faiss_manager.load_index(faiss_index_file)
    else:
        print("Processing artificial dataset...")
        chunked_artificial = data_processor.process_dataset(context_ds_artificial, is_labeled=True)
        print("Processing labeled dataset...")
        chunked_labeled = data_processor.process_dataset(context_ds_labeled, is_labeled=True)
        print("Processing unlabeled dataset...")
        chunked_unlabeled = data_processor.process_dataset(context_ds_unlabeled, is_labeled=False)
        print("Processing knowledge dataset...")
        chunked_knowledge = data_processor.process_dataset(context_ds_knowledge, is_labeled=False)

        # Combine all chunked data
        print("Combining all datasets...")
        all_chunked_data = chunked_artificial + chunked_labeled + chunked_unlabeled + chunked_knowledge

        # Generate and save embeddings
        embeddings = embedding_manager.generate_embeddings(all_chunked_data)
        embedding_manager.save_embeddings(embeddings, embeddings_file)

        # Build FAISS index
        faiss_manager.add_embeddings(embeddings)
        faiss_manager.save_index(faiss_index_file)

        # Generate metadata
        metadata = MetadataManager.generate_metadata(all_chunked_data)
        MetadataManager.save_metadata(metadata, metadata_file)

    # Perform query
    query_handler = QueryHandler(embedding_manager, faiss_manager, metadata)
    print("Querying dataset...")
    results = []
    sampled_questions = list(query_dataset)[:20]
    for sample in sampled_questions:
        question = sample["question"]
        options = [f"{key}. {value}" for key, value in sample["options"].items()]
        retrieved_contexts = query_handler.query_options_enhanced_with_keywords(
            question, options, top_k=5
        )
        results.append({"question": question, "options": sample["options"], "retrieved_contexts": retrieved_contexts})

    # Summarize results
    for result in results:
        print(f"Question: {result['question']}")
        for option, data in result['retrieved_contexts'].items():
            combined_text = " ".join([chunk["chunk_text"] for chunk in data['retrieved_chunks']])
            summary = summarizer.summarize(combined_text)
            print(f"Option: {option}\nSummary: {summary}\n")
