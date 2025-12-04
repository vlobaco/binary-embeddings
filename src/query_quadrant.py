import argparse
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from ollama import Client


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


def query_qdrant(
    text: str,
    collection_name: str,
    k: int = 5,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    ollama_host: str = "http://localhost:11434",
    model: str = "nomic-embed-text"
) -> List[Dict[str, Any]]:
    """
    Query Qdrant collection with the given text and return k most similar results.

    Args:
        text: Query text to search for
        collection_name: Name of the Qdrant collection to query
        k: Number of top results to return
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        ollama_host: Ollama server URL
        model: Embedding model to use

    Returns:
        List of dictionaries containing id, score, and payload for each result
    """
    # Initialize clients
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    ollama_client = Client(host=ollama_host)

    # Generate embedding for the query text
    print(f"Generating embedding for query text...")
    embedding = get_embedding(text, ollama_client, model)

    # Determine if we need binary embedding
    query_vector = embedding
    if "binary" in collection_name.lower():
        print(f"Converting to binary embedding...")
        query_vector = float_to_binary_embedding(embedding)

    # Query Qdrant
    print(f"Querying collection '{collection_name}' for top {k} results...")
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k
    ).points

    # Format results
    results = []
    for result in search_results:
        results.append({
            "id": result.id,
            "score": result.score,
        })

    return results


def print_results(results: List[Dict[str, Any]]):
    """
    Pretty print the query results
    """
    print(f"\n{'='*80}")
    print(f"Found {len(results)} results")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        print(f"Result #{i}")
        print(f"  ID: {result['id']}")
        print(f"  Score: {result['score']:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Query Qdrant collections with text and return k most similar results"
    )
    parser.add_argument(
        "text",
        type=str,
        help="Query text to search for"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="nq_float_embeddings",
        choices=["nq_float_embeddings", "nq_binary_embeddings"],
        help="Name of the Qdrant collection to query (default: nq_float_embeddings)"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant server host (default: localhost)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://rachel:11434",
        help="Ollama server URL (default: http://rachel:11434)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text",
        help="Embedding model to use (default: nomic-embed-text)"
    )
    
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Return only the IDs of results"
    )

    args = parser.parse_args()

    try:
        # Query Qdrant
        results = query_qdrant(
            text=args.text,
            collection_name=args.collection,
            k=args.k,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            ollama_host=args.ollama_host,
            model=args.model
        )

        # Print results
        if args.ids_only:
            print("\nResult IDs:")
            for result in results:
                print(result["id"])
        else:
            print_results(results)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
