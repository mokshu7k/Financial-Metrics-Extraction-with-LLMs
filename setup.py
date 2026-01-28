from setuptools import setup, find_packages

setup(
    name="financial-rag-research",
    version="1.0.0",
    description="Financial document processing and RAG evaluation pipeline",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "scipy>=1.8.0",
        "pdfplumber>=0.7.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.3",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.3",
            "torch>=1.12.0+cu116",
        ],
        "api": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "groq>=0.4.0"
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)