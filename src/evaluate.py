import json
import argparse
import random
import time
import os
import numpy as np
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from ollama import Client
import matplotlib.pyplot as plt
import seaborn as sns


def get_embedding(text: str, ollama_client: Client, model: str = "nomic-embed-text") -> List[float]:
    """
    Generate embedding for text using Ollama
    """
    response = ollama_client.embeddings(
        prompt=text,
        model=model
    )
    return response.embedding


def float_to_binary_embedding(embedding: List[float]) -> List[int]:
    """
    Convert float embedding to binary by checking if values are positive,
    then pack the bits into bytes using NumPy's packbits.
    Returns list of uint8 values (0-255), where each byte contains 8 bits.
    """
    binary_array = np.array([1 if x > 0 else 0 for x in embedding], dtype=np.uint8)
    packed_bytes = np.packbits(binary_array)
    return packed_bytes.tolist()


def calculate_ndcg(relevant_docs: List[int], retrieved_docs: List[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@k)
    relevant_docs: list of relevant document IDs (ground truth)
    retrieved_docs: list of retrieved document IDs in order
    """
    # Create relevance scores (1 if relevant, 0 if not)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            # Relevance is 1 for relevant docs
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1

    # Calculate ideal DCG (best possible ordering)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))

    # Return normalized DCG
    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    Returns 1/rank of first relevant document, 0 if none found
    """
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0


def calculate_precision_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """
    Calculate Precision@k
    """
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant_docs])
    return relevant_retrieved / k if k > 0 else 0.0


def calculate_recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """
    Calculate Recall@k
    """
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant_docs])
    return relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0.0


def search_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_vector: List,
    limit: int = 10
) -> Tuple[List[int], float]:
    """
    Search a collection and return retrieved document IDs and search time
    """
    start_time = time.time()
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    )
    search_time = time.time() - start_time

    retrieved_ids = [hit.id for hit in results.points]
    return retrieved_ids, search_time


def evaluate_question(
    question_idx: int,
    question_text: str,
    ollama_client: Client,
    qdrant_client: QdrantClient,
    k: int = 10
) -> Dict:
    """
    Evaluate a single question on both float and binary collections
    Returns metrics for both collections
    """
    # Generate query embedding
    embedding_start = time.time()
    query_embedding = get_embedding(question_text, ollama_client)
    embedding_time = time.time() - embedding_start

    # Convert to binary for binary search
    binary_query = float_to_binary_embedding(query_embedding)

    # Ground truth: the correct answer should be the same index as the question
    relevant_docs = [question_idx]

    # Search float collection
    float_retrieved, float_search_time = search_collection(
        qdrant_client,
        "nq_float_embeddings",
        query_embedding,
        limit=k
    )

    # Search binary collection
    binary_retrieved, binary_search_time = search_collection(
        qdrant_client,
        "nq_binary_embeddings",
        binary_query,
        limit=k
    )

    # Calculate metrics for float collection
    float_metrics = {
        "ndcg@10": calculate_ndcg(relevant_docs, float_retrieved, k=10),
        "ndcg@5": calculate_ndcg(relevant_docs, float_retrieved, k=5),
        "mrr": calculate_mrr(relevant_docs, float_retrieved),
        "precision@1": calculate_precision_at_k(relevant_docs, float_retrieved, k=1),
        "precision@5": calculate_precision_at_k(relevant_docs, float_retrieved, k=5),
        "recall@10": calculate_recall_at_k(relevant_docs, float_retrieved, k=10),
        "search_time": float_search_time,
        "total_time": embedding_time + float_search_time,
        "rank": float_retrieved.index(question_idx) + 1 if question_idx in float_retrieved else -1
    }

    # Calculate metrics for binary collection
    binary_metrics = {
        "ndcg@10": calculate_ndcg(relevant_docs, binary_retrieved, k=10),
        "ndcg@5": calculate_ndcg(relevant_docs, binary_retrieved, k=5),
        "mrr": calculate_mrr(relevant_docs, binary_retrieved),
        "precision@1": calculate_precision_at_k(relevant_docs, binary_retrieved, k=1),
        "precision@5": calculate_precision_at_k(relevant_docs, binary_retrieved, k=5),
        "recall@10": calculate_recall_at_k(relevant_docs, binary_retrieved, k=10),
        "search_time": binary_search_time,
        "total_time": embedding_time + binary_search_time,
        "rank": binary_retrieved.index(question_idx) + 1 if question_idx in binary_retrieved else -1
    }

    return {
        "question_index": question_idx,
        "question": question_text,
        "embedding_time": embedding_time,
        "float_metrics": float_metrics,
        "binary_metrics": binary_metrics
    }


def get_collection_size(qdrant_client: QdrantClient, collection_name: str) -> float:
    """
    Get the approximate size of a collection in MB
    """
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        # Get number of points and vector size
        points_count = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size

        # Estimate size: each float is 4 bytes, plus overhead
        if "binary" in collection_name:
            # Binary vectors are packed into bytes (8 bits per byte)
            bytes_per_vector = vector_size  # Already in bytes
        else:
            # Float vectors
            bytes_per_vector = vector_size * 4  # 4 bytes per float

        # Add some overhead for metadata and indexing (roughly 20%)
        total_bytes = points_count * bytes_per_vector * 1.2
        return total_bytes / (1024 * 1024)  # Convert to MB
    except Exception as e:
        print(f"Warning: Could not calculate size for {collection_name}: {e}")
        return 0.0


def create_comparison_plots(results: List[Dict], output_dir: str = "data/results", qdrant_client: QdrantClient = None):
    """
    Create visualization comparing float vs binary embeddings
    """
    print("\nGenerating comparison plots...")

    # Extract metrics
    questions = [r["question_index"] for r in results]

    # Metrics to compare (including time metrics)
    metric_names = ["ndcg@10", "ndcg@5", "mrr", "precision@1", "precision@5", "recall@10", "search_time"]

    # Set style
    sns.set_style("whitegrid")

    # Create subplots for each metric (4x2 grid, last slot will be empty)
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle("Float vs Binary Embeddings - Metric Comparison", fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metric_names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        float_values = [r["float_metrics"][metric] for r in results]
        binary_values = [r["binary_metrics"][metric] for r in results]

        # Convert time to milliseconds for better readability
        if "time" in metric:
            float_values = [v * 1000 for v in float_values]
            binary_values = [v * 1000 for v in binary_values]
            ylabel = f'{metric.replace("_", " ").title()} (ms)'
        else:
            ylabel = metric.upper()

        x = np.arange(len(questions))
        width = 0.35

        ax.bar(x - width/2, float_values, width, label='Float', alpha=0.8, color='#3498db')
        ax.bar(x + width/2, binary_values, width, label='Binary', alpha=0.8, color='#e74c3c')

        ax.set_xlabel('Question Index')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(questions)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide the last empty subplot
    axes[3, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/metrics_comparison.png")

    # Average metrics comparison (bar chart) - split into three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Average Metrics Comparison: Float vs Binary Embeddings', fontsize=14, fontweight='bold')

    # Quality metrics (left subplot)
    quality_metrics = ["ndcg@10", "ndcg@5", "mrr", "precision@1", "precision@5", "recall@10"]
    quality_labels = [m.replace("_", " ").upper() for m in quality_metrics]

    float_quality = []
    binary_quality = []

    for m in quality_metrics:
        float_vals = [r["float_metrics"][m] for r in results]
        binary_vals = [r["binary_metrics"][m] for r in results]
        float_quality.append(np.mean(float_vals))
        binary_quality.append(np.mean(binary_vals))

    x1 = np.arange(len(quality_labels))
    width = 0.35

    ax1.bar(x1 - width/2, float_quality, width, label='Float', alpha=0.8, color='#3498db')
    ax1.bar(x1 + width/2, binary_quality, width, label='Binary', alpha=0.8, color='#e74c3c')
    ax1.set_xlabel('Quality Metrics')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Quality Metrics')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(quality_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Time metrics (middle subplot) - only search time
    float_search = np.mean([r["float_metrics"]["search_time"] * 1000 for r in results])
    binary_search = np.mean([r["binary_metrics"]["search_time"] * 1000 for r in results])

    x2 = np.array([0])

    ax2.bar(x2 - width/2, [float_search], width, label='Float', alpha=0.8, color='#3498db')
    ax2.bar(x2 + width/2, [binary_search], width, label='Binary', alpha=0.8, color='#e74c3c')
    ax2.set_xlabel('Time Metric')
    ax2.set_ylabel('Average Time (ms)')
    ax2.set_title('Search Time Performance')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Search Time'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Disk space metrics (right subplot)
    if qdrant_client:
        float_size = get_collection_size(qdrant_client, "nq_float_embeddings")
        binary_size = get_collection_size(qdrant_client, "nq_binary_embeddings")

        x3 = np.array([0])

        ax3.bar(x3 - width/2, [float_size], width, label='Float', alpha=0.8, color='#3498db')
        ax3.bar(x3 + width/2, [binary_size], width, label='Binary', alpha=0.8, color='#e74c3c')
        ax3.set_xlabel('Storage Metric')
        ax3.set_ylabel('Disk Space (MB)')
        ax3.set_title('Index Storage Size')
        ax3.set_xticks(x3)
        ax3.set_xticklabels(['Disk Space'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Storage data unavailable', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Index Storage Size')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_metrics.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/average_metrics.png")

    plt.close('all')


def print_summary_statistics(results: List[Dict], float_size_mb: float = 0, binary_size_mb: float = 0):
    """
    Print summary statistics for the evaluation
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Calculate averages for float embeddings
    float_metrics = {
        "nDCG@10": np.mean([r["float_metrics"]["ndcg@10"] for r in results]),
        "nDCG@5": np.mean([r["float_metrics"]["ndcg@5"] for r in results]),
        "MRR": np.mean([r["float_metrics"]["mrr"] for r in results]),
        "Precision@1": np.mean([r["float_metrics"]["precision@1"] for r in results]),
        "Precision@5": np.mean([r["float_metrics"]["precision@5"] for r in results]),
        "Recall@10": np.mean([r["float_metrics"]["recall@10"] for r in results]),
        "Search Time (ms)": np.mean([r["float_metrics"]["search_time"] * 1000 for r in results]),
        "Disk Space (MB)": float_size_mb,
    }

    # Calculate averages for binary embeddings
    binary_metrics = {
        "nDCG@10": np.mean([r["binary_metrics"]["ndcg@10"] for r in results]),
        "nDCG@5": np.mean([r["binary_metrics"]["ndcg@5"] for r in results]),
        "MRR": np.mean([r["binary_metrics"]["mrr"] for r in results]),
        "Precision@1": np.mean([r["binary_metrics"]["precision@1"] for r in results]),
        "Precision@5": np.mean([r["binary_metrics"]["precision@5"] for r in results]),
        "Recall@10": np.mean([r["binary_metrics"]["recall@10"] for r in results]),
        "Search Time (ms)": np.mean([r["binary_metrics"]["search_time"] * 1000 for r in results]),
        "Disk Space (MB)": binary_size_mb,
    }

    print("\nFLOAT EMBEDDINGS:")
    print("-" * 40)
    for metric, value in float_metrics.items():
        print(f"  {metric:20s}: {value:8.4f}")

    print("\nBINARY EMBEDDINGS:")
    print("-" * 40)
    for metric, value in binary_metrics.items():
        print(f"  {metric:20s}: {value:8.4f}")

    print("\nSPEEDUP FACTOR:")
    print("-" * 40)
    speedup_search = float_metrics["Search Time (ms)"] / binary_metrics["Search Time (ms)"]
    print(f"  Search Time       : {speedup_search:.2f}x")

    print("\nSTORAGE REDUCTION:")
    print("-" * 40)
    if float_size_mb > 0 and binary_size_mb > 0:
        size_ratio = float_size_mb / binary_size_mb
        savings_mb = float_size_mb - binary_size_mb
        savings_percent = (savings_mb / float_size_mb) * 100
        print(f"  Size Ratio        : {size_ratio:.2f}x")
        print(f"  Space Saved (MB)  : {savings_mb:.2f}")
        print(f"  Space Saved (%)   : {savings_percent:.2f}%")
    else:
        print("  Data unavailable")

    print("\nMETRIC DEGRADATION:")
    print("-" * 40)
    for key in ["nDCG@10", "nDCG@5", "MRR", "Precision@1", "Precision@5", "Recall@10"]:
        degradation = (float_metrics[key] - binary_metrics[key]) / float_metrics[key] * 100 if float_metrics[key] > 0 else 0
        print(f"  {key:20s}: {degradation:+6.2f}%")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate float vs binary embeddings on test dataset")
    parser.add_argument("-k", "--num-questions", type=int, default=10,
                      help="Number of random questions to evaluate (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                      help="Generate comparison plots and detailed visualizations")
    parser.add_argument("--test-file", type=str, default="data/test_dataset.jsonl",
                      help="Path to test dataset file (default: data/test_dataset.jsonl)")
    parser.add_argument("--output-dir", type=str, default="data/results",
                      help="Directory to save results and plots (default: data/results)")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("EMBEDDING EVALUATION: Float vs Binary")
    print(f"{'='*80}\n")

    # Load test dataset
    print(f"Loading test dataset from {args.test_file}...")
    test_questions = []
    with open(args.test_file, "r") as f:
        for line in f:
            if line.strip():
                test_questions.append(json.loads(line))

    print(f"  Loaded {len(test_questions)} questions")

    # Select k random questions
    k = min(args.num_questions, len(test_questions))
    selected_questions = random.sample(test_questions, k)
    print(f"  Selected {k} random questions for evaluation")

    # Connect to services
    print("\nConnecting to services...")
    qdrant_client = QdrantClient(host="localhost", port=6333)
    ollama_client = Client(host="http://rachel:11434")
    print("  Connected to Qdrant and Ollama")

    # Run evaluation
    print(f"\nEvaluating {k} questions...")
    results = []
    for q in tqdm(selected_questions, desc="Evaluating questions", unit="question", colour="blue"):
        result = evaluate_question(
            q["index"],
            q["question"],
            ollama_client,
            qdrant_client,
            k=10
        )
        results.append(result)

    # Calculate summary statistics
    print("\nCalculating summary statistics...")

    # Calculate disk space
    float_size_mb = get_collection_size(qdrant_client, "nq_float_embeddings")
    binary_size_mb = get_collection_size(qdrant_client, "nq_binary_embeddings")

    # Aggregate metrics
    summary = {
        "total_questions": len(results),
        "float_embeddings": {
            "ndcg@10": np.mean([r["float_metrics"]["ndcg@10"] for r in results]),
            "ndcg@5": np.mean([r["float_metrics"]["ndcg@5"] for r in results]),
            "mrr": np.mean([r["float_metrics"]["mrr"] for r in results]),
            "precision@1": np.mean([r["float_metrics"]["precision@1"] for r in results]),
            "precision@5": np.mean([r["float_metrics"]["precision@5"] for r in results]),
            "recall@10": np.mean([r["float_metrics"]["recall@10"] for r in results]),
            "search_time_ms": np.mean([r["float_metrics"]["search_time"] * 1000 for r in results]),
            "disk_space_mb": float_size_mb,
        },
        "binary_embeddings": {
            "ndcg@10": np.mean([r["binary_metrics"]["ndcg@10"] for r in results]),
            "ndcg@5": np.mean([r["binary_metrics"]["ndcg@5"] for r in results]),
            "mrr": np.mean([r["binary_metrics"]["mrr"] for r in results]),
            "precision@1": np.mean([r["binary_metrics"]["precision@1"] for r in results]),
            "precision@5": np.mean([r["binary_metrics"]["precision@5"] for r in results]),
            "recall@10": np.mean([r["binary_metrics"]["recall@10"] for r in results]),
            "search_time_ms": np.mean([r["binary_metrics"]["search_time"] * 1000 for r in results]),
            "disk_space_mb": binary_size_mb,
        },
        "speedup": {
            "search_time": np.mean([r["float_metrics"]["search_time"] for r in results]) / np.mean([r["binary_metrics"]["search_time"] for r in results]),
        },
        "space_reduction": {
            "size_ratio": float_size_mb / binary_size_mb if binary_size_mb > 0 else 0,
            "savings_mb": float_size_mb - binary_size_mb,
            "savings_percent": ((float_size_mb - binary_size_mb) / float_size_mb * 100) if float_size_mb > 0 else 0,
        },
        "metric_degradation_percent": {
            "ndcg@10": (np.mean([r["float_metrics"]["ndcg@10"] for r in results]) - np.mean([r["binary_metrics"]["ndcg@10"] for r in results])) / np.mean([r["float_metrics"]["ndcg@10"] for r in results]) * 100,
            "ndcg@5": (np.mean([r["float_metrics"]["ndcg@5"] for r in results]) - np.mean([r["binary_metrics"]["ndcg@5"] for r in results])) / np.mean([r["float_metrics"]["ndcg@5"] for r in results]) * 100,
            "mrr": (np.mean([r["float_metrics"]["mrr"] for r in results]) - np.mean([r["binary_metrics"]["mrr"] for r in results])) / np.mean([r["float_metrics"]["mrr"] for r in results]) * 100,
            "precision@1": (np.mean([r["float_metrics"]["precision@1"] for r in results]) - np.mean([r["binary_metrics"]["precision@1"] for r in results])) / np.mean([r["float_metrics"]["precision@1"] for r in results]) * 100,
        }
    }

    # Save summary to JSON
    summary_file = f"{args.output_dir}/summary.json"
    print(f"\nSaving summary to {summary_file}...")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary statistics")

    # Print summary statistics
    print_summary_statistics(results, float_size_mb=float_size_mb, binary_size_mb=binary_size_mb)

    # Generate plots if verbose mode
    if args.verbose:
        create_comparison_plots(results, output_dir=args.output_dir, qdrant_client=qdrant_client)
        print("\nVisualization complete!")

    print(f"\nEvaluation complete! Summary saved to {summary_file}")
    if args.verbose:
        print(f"Plots saved to {args.output_dir}/ directory")


if __name__ == "__main__":
    main()
