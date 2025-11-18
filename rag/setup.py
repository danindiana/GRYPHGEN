#!/usr/bin/env python3
"""
Setup script for GRYPHGEN RAG package.
Modern implementation of SimGRAG and CAG.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="gryphgen-rag",
    version="1.0.0",
    description="Modern implementation of SimGRAG and CAG: Advanced Retrieval-Augmented Generation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GRYPHGEN Contributors",
    author_email="info@gryphgen.ai",
    url="https://github.com/danindiana/GRYPHGEN",
    project_urls={
        "Documentation": "https://github.com/danindiana/GRYPHGEN/tree/main/rag",
        "Source": "https://github.com/danindiana/GRYPHGEN",
        "Tracker": "https://github.com/danindiana/GRYPHGEN/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "transformers>=4.47.0",
        "sentence-transformers>=3.3.0",
        "networkx>=3.4.2",
        "ollama>=0.4.4",
        "sqlalchemy>=2.0.36",
        "numpy>=2.1.3",
        "pydantic>=2.10.3",
        "pyyaml>=6.0.2",
        "tqdm>=4.67.1",
        "rich>=13.9.4",
        "loguru>=0.7.3",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.4",
            "pytest-cov>=6.0.0",
            "black>=24.10.0",
            "ruff>=0.8.4",
            "mypy>=1.13.0",
            "pre-commit>=4.0.1",
        ],
        "docs": [
            "mkdocs>=1.6.1",
            "mkdocs-material>=9.5.48",
            "mkdocstrings>=0.27.0",
        ],
        "gpu": [
            "faiss-gpu>=1.9.0",
            "cupy-cuda12x>=13.3.0",
            "triton>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gryphgen-rag=gryphgen_rag.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "knowledge-graphs",
        "llm",
        "ai",
        "machine-learning",
        "simgrag",
        "cag",
        "cache-augmented-generation",
        "nvidia",
        "gpu",
    ],
    include_package_data=True,
    zip_safe=False,
)
