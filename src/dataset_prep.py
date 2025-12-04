import json
import numpy as np
import argparse
from typing import List
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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
    # Convert to binary array (0 or 1)
    binary_array = np.array([1 if x > 0 else 0 for x in embedding], dtype=np.uint8)

    # Pack bits into bytes (8 bits per byte)
    packed_bytes = np.packbits(binary_array)

    # Convert to Python list for Qdrant
    return packed_bytes.tolist()

def setup_collections(
    client: QdrantClient,
    vector_size: int = 768,
    float_collection: str = "nq_float_embeddings",
    binary_collection: str = "nq_binary_embeddings"
):
    """
    Create two collections: one for float embeddings, one for binary embeddings.
    nomic-embed-text produces 768-dimensional vectors.
    Binary embeddings are packed into bytes (vector_size / 8).
    """
    # Collection for regular float embeddings (using cosine similarity)
    if client.collection_exists(float_collection):
        client.delete_collection(float_collection)
    client.create_collection(
        collection_name=float_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    # Collection for binary embeddings (using Manhattan distance)
    # Binary vectors are packed into bytes, so size is vector_size / 8
    binary_vector_size = vector_size // 8
    if client.collection_exists(binary_collection):
        client.delete_collection(binary_collection)
    client.create_collection(
        collection_name=binary_collection,
        vectors_config=VectorParams(size=binary_vector_size, distance=Distance.MANHATTAN)
    )

    print(f"Created collections: float vectors ({vector_size}D), binary vectors ({binary_vector_size} bytes)")

def store_entity_with_embeddings(
    client: QdrantClient,
    id: int,
    embedding: List[float],
    document_text: str,
    question_text: str,
    float_collection: str = "nq_float_embeddings",
    binary_collection: str = "nq_binary_embeddings"
):
    """
    Store entity in both collections with float and binary embeddings.
    Stores document_text and question_text in the payload for retrieval.
    """
    binary_embedding = float_to_binary_embedding(embedding)

    # Store in float collection
    client.upsert(
        collection_name=float_collection,
        points=[
            PointStruct(
                id=id,
                vector=embedding
            )
        ]
    )

    # Store in binary collection
    client.upsert(
        collection_name=binary_collection,
        points=[
            PointStruct(
                id=id,
                vector=binary_embedding
            )
        ]
    )

# Main execution
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Prepare dataset with float and binary embeddings for Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="data/simplified-nq-train.jsonl",
        help="Path to input JSONL file containing the dataset"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/test_dataset.jsonl",
        help="Path to output test dataset JSONL file"
    )

    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant server host"
    )

    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port"
    )

    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://rachel:11434",
        help="Ollama server URL"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model to use"
    )

    parser.add_argument(
        "--vector-size",
        type=int,
        default=768,
        help="Size of the embedding vectors"
    )

    parser.add_argument(
        "--max-entities",
        type=int,
        default=10,
        help="Maximum number of entities to process (use -1 for unlimited)"
    )

    parser.add_argument(
        "--float-collection",
        type=str,
        default="nq_float_embeddings",
        help="Name for the float embeddings collection"
    )

    parser.add_argument(
        "--binary-collection",
        type=str,
        default="nq_binary_embeddings",
        help="Name for the binary embeddings collection"
    )

    args = parser.parse_args()

    # Connect to Qdrant
    qdrant_client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    # Connect Ollama
    ollama_client = Client(host=args.ollama_host)

    print("Setting up Qdrant collections...")
    setup_collections(
        qdrant_client,
        vector_size=args.vector_size,
        float_collection=args.float_collection,
        binary_collection=args.binary_collection
    )

    # Count total lines for progress bar
    print("\nCounting entities...")
    with open(args.input_path, "r") as file:
        total_lines = sum(1 for _ in file)

    # Determine actual number of entities to process
    if args.max_entities == -1:
        total_to_process = total_lines
    else:
        total_to_process = min(args.max_entities, total_lines)

    print(f"Processing {total_to_process} entities and generating embeddings...")

    # Process entities from JSONL file
    entity_count = 0
    test_dataset = []

    with open(args.input_path, "r") as file:
        with tqdm(total=total_to_process, desc="Processing entities", unit="entity", colour="blue") as pbar:
            for line in file:
                if args.max_entities != -1 and entity_count >= args.max_entities:
                    break

                object = json.loads(line)
                document_text = object["document_text"]
                question_text = object["question_text"]

                # Generate embedding
                embedding = get_embedding(document_text, ollama_client, model=args.embedding_model)

                # Store in both collections
                store_entity_with_embeddings(
                    qdrant_client,
                    entity_count,
                    embedding,
                    document_text,
                    question_text,
                    float_collection=args.float_collection,
                    binary_collection=args.binary_collection
                )

                # Add to test dataset
                test_dataset.append({
                    "index": entity_count,
                    "question": question_text
                })

                entity_count += 1
                pbar.update(1)

    # Write test dataset to JSONL file
    print(f"\nWriting test dataset to {args.output_path}...")
    with open(args.output_path, "w") as test_file:
        for entry in test_dataset:
            test_file.write(json.dumps(entry) + "\n")

    print(f"\n✓ Successfully processed and stored {entity_count} entities in Qdrant!")
    print(f"✓ Test dataset with {len(test_dataset)} questions saved to {args.output_path}")
    print("\nCollections created:")
    print(f"  - {args.float_collection}: Regular float embeddings with cosine similarity")
    print(f"  - {args.binary_collection}: Binary embeddings with Manhattan distance")
