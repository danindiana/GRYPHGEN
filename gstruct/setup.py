"""Setup script for GRYPHGEN Grid Computing Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="gryphgen",
    version="2.0.0",
    description="Advanced Grid Computing Framework with LLM-based Orchestration and GPU Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GRYPHGEN Team",
    author_email="gryphgen@example.com",
    url="https://github.com/danindiana/GRYPHGEN",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    python_requires=">=3.11",
    install_requires=[
        line.strip().split(">=")[0]
        for line in Path("requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#") and ">=" in line
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.5",
            "pytest-cov>=4.1.0",
            "black>=24.2.0",
            "ruff>=0.2.2",
            "mypy>=1.8.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gryphgen=GRYPHGEN:main",
            "gryphgen-orchestrate=SYMORQ.orchestration:main",
            "gryphgen-schedule=SYMORG.scheduling:main",
            "gryphgen-deploy=SYMAUG.scripts.deployment:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="grid-computing distributed-systems gpu cuda llm orchestration scheduling",
    project_urls={
        "Bug Reports": "https://github.com/danindiana/GRYPHGEN/issues",
        "Source": "https://github.com/danindiana/GRYPHGEN",
        "Documentation": "https://github.com/danindiana/GRYPHGEN/tree/main/gstruct/docs",
    },
)
