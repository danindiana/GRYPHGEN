#!/usr/bin/env python3
"""
Initialize FAISS database with sample code examples.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calisota.core.config import get_settings
from src.calisota.rag.faiss_manager import FAISSManager


def main() -> None:
    """Initialize FAISS with sample data."""
    print("Initializing FAISS database...")

    settings = get_settings()
    faiss = FAISSManager(settings)

    # Initialize index
    faiss.initialize_index(force_new=True)

    # Sample code examples for different languages
    sample_texts = [
        # Python examples
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "def binary_search(arr, target): low, high = 0, len(arr)-1; while low <= high: mid = (low + high) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: low = mid + 1; else: high = mid - 1; return -1",
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])",

        # Rust examples
        "fn fibonacci(n: u64) -> u64 { match n { 0 | 1 => n, _ => fibonacci(n-1) + fibonacci(n-2) } }",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> { let mut low = 0; let mut high = arr.len(); while low < high { let mid = (low + high) / 2; match arr[mid].cmp(&target) { std::cmp::Ordering::Equal => return Some(mid), std::cmp::Ordering::Less => low = mid + 1, std::cmp::Ordering::Greater => high = mid, } } None }",

        # Go examples
        "func fibonacci(n int) int { if n <= 1 { return n }; return fibonacci(n-1) + fibonacci(n-2) }",
        "func binarySearch(arr []int, target int) int { low, high := 0, len(arr)-1; for low <= high { mid := (low + high) / 2; if arr[mid] == target { return mid } else if arr[mid] < target { low = mid + 1 } else { high = mid - 1 } }; return -1 }",

        # Documentation and concepts
        "Fibonacci sequence: A series where each number is the sum of the two preceding ones",
        "Binary search: An efficient algorithm for finding an item in a sorted list with O(log n) time complexity",
        "Quicksort: A divide-and-conquer sorting algorithm with O(n log n) average time complexity",
        "Recursion: A programming technique where a function calls itself to solve smaller instances of the same problem",
        "Dynamic programming: An optimization technique that stores results of subproblems to avoid redundant calculations",

        # Best practices
        "Always validate input parameters before processing",
        "Use type hints in Python for better code documentation and IDE support",
        "Implement error handling with try-except blocks for robust code",
        "Write unit tests for all critical functions",
        "Use async/await for I/O-bound operations to improve performance",
    ]

    metadata = [
        {"language": "python", "topic": "recursion", "algorithm": "fibonacci"},
        {"language": "python", "topic": "searching", "algorithm": "binary_search"},
        {"language": "python", "topic": "sorting", "algorithm": "quicksort"},
        {"language": "rust", "topic": "recursion", "algorithm": "fibonacci"},
        {"language": "rust", "topic": "searching", "algorithm": "binary_search"},
        {"language": "go", "topic": "recursion", "algorithm": "fibonacci"},
        {"language": "go", "topic": "searching", "algorithm": "binary_search"},
        {"language": "general", "topic": "concepts", "algorithm": "fibonacci"},
        {"language": "general", "topic": "concepts", "algorithm": "binary_search"},
        {"language": "general", "topic": "concepts", "algorithm": "quicksort"},
        {"language": "general", "topic": "concepts", "algorithm": "recursion"},
        {"language": "general", "topic": "concepts", "algorithm": "dynamic_programming"},
        {"language": "general", "topic": "best_practices"},
        {"language": "python", "topic": "best_practices"},
        {"language": "general", "topic": "best_practices"},
        {"language": "general", "topic": "best_practices"},
        {"language": "python", "topic": "best_practices"},
    ]

    # Add to FAISS
    print(f"Adding {len(sample_texts)} sample documents...")
    faiss.add_embeddings(sample_texts, metadata)

    # Save index
    print("Saving index...")
    faiss.save_index()

    # Test search
    print("\nTesting search...")
    results = faiss.search("how to implement fibonacci in python", top_k=3)
    print(f"\nSearch results for 'fibonacci in python':")
    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result['similarity']:.3f}] {result.get('text', 'N/A')[:80]}...")

    print(f"\n✅ FAISS database initialized successfully!")
    print(f"Total vectors: {faiss.index.ntotal}")
    print(f"Index path: {settings.faiss_index_path}")


if __name__ == "__main__":
    main()
