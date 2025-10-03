# src/data_preprocessing/embed_store.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, asdict
import pickle
import sqlite3
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_type: str = "faiss"  # "faiss", "chroma", or "sqlite"
    index_type: str = "flat"   # "flat", "ivf", "hnsw"
    similarity_metric: str = "cosine"  # "cosine", "dot", "euclidean"
    persist_directory: str = "data/embeddings"

class FinancialEmbeddingGenerator:
    """
    Generates embeddings for financial document chunks using various models.
    Supports both local models (sentence-transformers) and API-based models.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
            self.logger.info(f"Loaded embedding model: {self.config.model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model {self.config.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            return embeddings
        
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]
    
    def embed_chunks_from_file(self, chunks_file_path: Path) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks loaded from a JSON file."""
        try:
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            texts = [chunk['content'] for chunk in chunks_data]
            self.logger.info(f"Generating embeddings for {len(texts)} chunks from {chunks_file_path.name}")
            
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to chunk data
            embedded_chunks = []
            for i, chunk in enumerate(chunks_data):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embeddings[i].tolist()
                chunk_with_embedding['embedding_model'] = self.config.model_name
                chunk_with_embedding['embedding_timestamp'] = datetime.now().isoformat()
                embedded_chunks.append(chunk_with_embedding)
            
            return embedded_chunks
            
        except Exception as e:
            self.logger.error(f"Error embedding chunks from {chunks_file_path}: {e}")
            raise

class FAISSVectorStore:
    """FAISS-based vector store for financial document embeddings."""
    
    def __init__(self, config: VectorStoreConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.config = config
        self.index = None
        self.metadata_db = None
        self.dimension = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure persist directory exists
        Path(config.persist_directory).mkdir(parents=True, exist_ok=True)
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.config.index_type == "flat":
            if self.config.similarity_metric == "cosine":
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            else:
                index = faiss.IndexFlatL2(dimension)  # L2 distance
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        else:
            # Default to flat index
            index = faiss.IndexFlatL2(dimension)
        
        return index
    
    def _setup_metadata_db(self):
        """Setup SQLite database for metadata storage."""
        db_path = Path(self.config.persist_directory) / "metadata.db"
        self.metadata_db = sqlite3.connect(str(db_path))
        
        # Create metadata table
        self.metadata_db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                document_id TEXT,
                company TEXT,
                quarter TEXT,
                year INTEGER,
                document_type TEXT,
                chunk_type TEXT,
                section_type TEXT,
                contains_financial_data INTEGER,
                token_count INTEGER,
                content TEXT
            )
        """)
        self.metadata_db.commit()
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add embeddings and metadata to the vector store."""
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = self._create_index(self.dimension)
            self._setup_metadata_db()
        
        # Normalize embeddings for cosine similarity if needed
        if self.config.similarity_metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Add metadata to database
        for i, metadata in enumerate(metadata_list):
            self.metadata_db.execute("""
                INSERT OR REPLACE INTO chunk_metadata 
                (chunk_id, document_id, company, quarter, year, document_type, 
                 chunk_type, section_type, contains_financial_data, token_count, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['metadata']['chunk_id'],
                metadata['metadata']['document_id'],
                metadata['metadata']['company'],
                metadata['metadata']['quarter'],
                metadata['metadata']['year'],
                metadata['metadata']['document_type'],
                metadata['metadata']['chunk_type'],
                metadata['metadata'].get('section_type'),
                int(metadata['metadata']['contains_financial_data']),
                metadata['metadata']['token_count'],
                metadata['content']
            ))
        
        self.metadata_db.commit()
        self.logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        if self.index is None:
            raise RuntimeError("Index not initialized. Add embeddings first.")
        
        # Normalize query embedding if using cosine similarity
        if self.config.similarity_metric == "cosine":
            query_embedding = query_embedding.copy()
            faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        # Retrieve metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            # Get metadata from database
            cursor = self.metadata_db.execute(
                "SELECT * FROM chunk_metadata WHERE rowid = ?", (int(idx) + 1,)
            )
            row = cursor.fetchone()
            
            if row:
                result = {
                    'score': float(score),
                    'rank': i + 1,
                    'chunk_id': row[1],
                    'document_id': row[2],
                    'company': row[3],
                    'quarter': row[4],
                    'year': row[5],
                    'document_type': row[6],
                    'chunk_type': row[7],
                    'section_type': row[8],
                    'contains_financial_data': bool(row[9]),
                    'token_count': row[10],
                    'content': row[11]
                }
                
                # Apply filters if specified
                if filter_criteria:
                    if self._matches_filter(result, filter_criteria):
                        results.append(result)
                else:
                    results.append(result)
        
        return results
    
    def _matches_filter(self, result: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if result matches filter criteria."""
        for key, value in filter_criteria.items():
            if key in result:
                if isinstance(value, list):
                    if result[key] not in value:
                        return False
                else:
                    if result[key] != value:
                        return False
        return True
    
    def save_index(self, filename: str = "financial_index.faiss"):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        index_path = Path(self.config.persist_directory) / filename
        faiss.write_index(self.index, str(index_path))
        self.logger.info(f"Saved FAISS index to {index_path}")
    
    def load_index(self, filename: str = "financial_index.faiss"):
        """Load FAISS index from disk."""
        index_path = Path(self.config.persist_directory) / filename
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        self._setup_metadata_db()
        self.logger.info(f"Loaded FAISS index from {index_path}")

class ChromaVectorStore:
    """ChromaDB-based vector store for financial document embeddings."""
    
    def __init__(self, config: VectorStoreConfig):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
        
        self.config = config
        self.client = None
        self.collection = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        settings = Settings(
            persist_directory=self.config.persist_directory,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="financial_chunks",
            metadata={"description": "Financial document chunks for RAG"}
        )
        
        self.logger.info("Initialized ChromaDB client")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add embeddings and metadata to ChromaDB."""
        chunk_ids = [chunk['metadata']['chunk_id'] for chunk in metadata_list]
        documents = [chunk['content'] for chunk in metadata_list]
        
        # Prepare metadata for ChromaDB (flatten nested structure)
        chroma_metadata = []
        for chunk in metadata_list:
            meta = chunk['metadata'].copy()
            # Convert nested objects to strings if needed
            for key, value in meta.items():
                if isinstance(value, (dict, list)):
                    meta[key] = str(value)
            chroma_metadata.append(meta)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=chroma_metadata,
            ids=chunk_ids
        )
        
        self.logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB."""
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_criteria
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'rank': i + 1,
                'chunk_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                **results['metadatas'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results

class FinancialVectorStoreManager:
    """
    Manages the creation and population of vector stores for financial documents.
    Supports multiple vector store backends (FAISS, ChromaDB).
    """
    
    def __init__(self, 
                 embedding_config: EmbeddingConfig = None,
                 vector_config: VectorStoreConfig = None,
                 chunks_dir: Path = Path("data/processed/chunks")):
        
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_config = vector_config or VectorStoreConfig()
        self.chunks_dir = chunks_dir
        
        # Initialize components
        self.embedding_generator = FinancialEmbeddingGenerator(self.embedding_config)
        
        if self.vector_config.store_type == "faiss":
            self.vector_store = FAISSVectorStore(self.vector_config)
        elif self.vector_config.store_type == "chroma":
            self.vector_store = ChromaVectorStore(self.vector_config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_config.store_type}")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def create_vector_store_from_chunks(self, chunk_type: str = "fixed_size"):
        """Create vector store from processed document chunks."""
        
        # Find chunk files
        financial_chunks_file = self.chunks_dir / f"financial_chunks_{chunk_type}.json"
        transcript_chunks_file = self.chunks_dir / f"transcript_chunks_{chunk_type}.json"
        
        all_chunks = []
        
        # Process financial chunks
        if financial_chunks_file.exists():
            financial_chunks = self.embedding_generator.embed_chunks_from_file(financial_chunks_file)
            all_chunks.extend(financial_chunks)
            self.logger.info(f"Processed {len(financial_chunks)} financial chunks")
        
        # Process transcript chunks
        if transcript_chunks_file.exists():
            transcript_chunks = self.embedding_generator.embed_chunks_from_file(transcript_chunks_file)
            all_chunks.extend(transcript_chunks)
            self.logger.info(f"Processed {len(transcript_chunks)} transcript chunks")
        
        if not all_chunks:
            raise RuntimeError("No chunk files found. Run chunking process first.")
        
        # Extract embeddings and metadata
        embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
        
        # Add embeddings to vector store
        self.vector_store.add_embeddings(embeddings, all_chunks)
        
        # Save vector store
        if hasattr(self.vector_store, 'save_index'):
            self.vector_store.save_index()
        
        # Save embedded chunks for future use
        output_path = Path(self.vector_config.persist_directory) / f"embedded_chunks_{chunk_type}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        
        self.logger.info(f"Created vector store with {len(all_chunks)} chunks")
        return len(all_chunks)
    
    def search_similar_chunks(self, query: str, k: int = 10, 
                            filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for chunks similar to a query."""
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k, filter_criteria)
        
        return results
    
    def evaluate_retrieval_quality(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate retrieval quality using test queries."""
        
        metrics = {
            'precision_at_5': 0.0,
            'recall_at_10': 0.0,
            'mrr': 0.0  # Mean Reciprocal Rank
        }
        
        if not test_queries:
            return metrics
        
        total_precision = 0
        total_recall = 0
        total_mrr = 0
        
        for query_data in test_queries:
            query = query_data['query']
            expected_companies = query_data.get('expected_companies', [])
            
            results = self.search_similar_chunks(query, k=10)
            
            # Calculate precision@5
            top_5_companies = [r['company'] for r in results[:5]]
            relevant_in_top5 = len(set(top_5_companies) & set(expected_companies))
            precision_5 = relevant_in_top5 / min(5, len(expected_companies))
            
            # Calculate recall@10
            top_10_companies = [r['company'] for r in results[:10]]
            relevant_in_top10 = len(set(top_10_companies) & set(expected_companies))
            recall_10 = relevant_in_top10 / len(expected_companies) if expected_companies else 0
            
            # Calculate MRR
            mrr_score = 0
            for i, result in enumerate(results):
                if result['company'] in expected_companies:
                    mrr_score = 1 / (i + 1)
                    break
            
            total_precision += precision_5
            total_recall += recall_10
            total_mrr += mrr_score
        
        num_queries = len(test_queries)
        metrics['precision_at_5'] = total_precision / num_queries
        metrics['recall_at_10'] = total_recall / num_queries
        metrics['mrr'] = total_mrr / num_queries
        
        return metrics
    
    def generate_sample_queries(self) -> List[Dict[str, Any]]:
        """Generate sample queries for testing retrieval quality."""
        return [
            {
                'query': 'What was the revenue growth in Q1 2024?',
                'expected_companies': ['BRITANNIA'],
                'expected_document_type': 'financial'
            },
            {
                'query': 'Tell me about management discussion on market expansion',
                'expected_companies': ['BRITANNIA'],
                'expected_document_type': 'transcript'
            },
            {
                'query': 'What are the key financial metrics for the quarter?',
                'expected_companies': ['BRITANNIA'],
                'expected_document_type': 'financial'
            }
        ]

def create_financial_vector_store(
    chunks_dir: Path = Path("data/processed/chunks"),
    embeddings_dir: Path = Path("data/embeddings"),
    embedding_model: str = "all-MiniLM-L6-v2",
    vector_store_type: str = "faiss",
    chunk_type: str = "fixed_size"
) -> FinancialVectorStoreManager:
    """
    Convenience function to create a financial vector store with default settings.
    
    Args:
        chunks_dir: Directory containing processed chunks
        embeddings_dir: Directory to save embeddings and vector store
        embedding_model: Name of the embedding model to use
        vector_store_type: Type of vector store ("faiss" or "chroma")
        chunk_type: Type of chunks to use ("fixed_size", "paragraph", "section")
    
    Returns:
        Configured FinancialVectorStoreManager
    """
    
    embedding_config = EmbeddingConfig(
        model_name=embedding_model,
        dimension=384 if "MiniLM" in embedding_model else 768,
        normalize_embeddings=True,
        batch_size=32
    )
    
    vector_config = VectorStoreConfig(
        store_type=vector_store_type,
        index_type="flat",
        similarity_metric="cosine",
        persist_directory=str(embeddings_dir)
    )
    
    manager = FinancialVectorStoreManager(
        embedding_config=embedding_config,
        vector_config=vector_config,
        chunks_dir=chunks_dir
    )
    
    return manager

def benchmark_embedding_models(chunks_sample: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark different embedding models on a sample of financial text.
    
    Args:
        chunks_sample: Sample of text chunks for benchmarking
    
    Returns:
        Dictionary with benchmark results for each model
    """
    
    models_to_test = [
        "all-MiniLM-L6-v2",           # Fast, lightweight
        "all-mpnet-base-v2",          # Better quality, slower
        "sentence-transformers/all-MiniLM-L12-v2",  # Balance of speed/quality
    ]
    
    benchmark_results = {}
    
    for model_name in models_to_test:
        try:
            config = EmbeddingConfig(model_name=model_name)
            generator = FinancialEmbeddingGenerator(config)
            
            # Time embedding generation
            import time
            start_time = time.time()
            embeddings = generator.generate_embeddings(chunks_sample[:100])  # Test on first 100 chunks
            end_time = time.time()
            
            benchmark_results[model_name] = {
                'embedding_time': end_time - start_time,
                'embeddings_per_second': len(chunks_sample[:100]) / (end_time - start_time),
                'embedding_dimension': embeddings.shape[1],
                'model_size_mb': 'Unknown',  # Would need to implement model size detection
                'suitable_for': 'production' if 'MiniLM' in model_name else 'research'
            }
            
        except Exception as e:
            benchmark_results[model_name] = {'error': str(e)}
    
    return benchmark_results

# Example usage and testing functions
def test_vector_store_functionality():
    """Test the vector store functionality with sample data."""
    
    print("Testing Vector Store Functionality")
    print("=" * 50)
    
    try:
        # Create vector store manager
        manager = create_financial_vector_store(
            chunks_dir=Path("data/processed/chunks"),
            embeddings_dir=Path("data/embeddings"),
            embedding_model="all-MiniLM-L6-v2",
            vector_store_type="faiss"
        )
        
        # Create vector store from chunks
        num_chunks = manager.create_vector_store_from_chunks("fixed_size")
        print(f"Created vector store with {num_chunks} chunks")
        
        # Test search functionality
        test_queries = [
            "revenue growth in the quarter",
            "management discussion about future outlook",
            "financial metrics and profitability"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = manager.search_similar_chunks(query, k=3)
            
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['company']} {result['quarter']} {result['year']} "
                      f"(Score: {result['score']:.3f})")
                print(f"     {result['content'][:100]}...")
        
        # Evaluate retrieval quality
        sample_queries = manager.generate_sample_queries()
        metrics = manager.evaluate_retrieval_quality(sample_queries)
        
        print(f"\nRetrieval Quality Metrics:")
        print(f"  Precision@5: {metrics['precision_at_5']:.3f}")
        print(f"  Recall@10: {metrics['recall_at_10']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def generate_vector_store_report(manager: FinancialVectorStoreManager) -> Dict[str, Any]:
    """Generate a comprehensive report about the vector store."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'embedding_model': manager.embedding_config.model_name,
            'embedding_dimension': manager.embedding_config.dimension,
            'vector_store_type': manager.vector_config.store_type,
            'similarity_metric': manager.vector_config.similarity_metric
        },
        'statistics': {},
        'sample_searches': [],
        'recommendations': []
    }
    
    # Add sample search results
    sample_queries = [
        "quarterly revenue performance",
        "management outlook and guidance",
        "cost reduction and efficiency measures"
    ]
    
    for query in sample_queries:
        try:
            results = manager.search_similar_chunks(query, k=5)
            report['sample_searches'].append({
                'query': query,
                'num_results': len(results),
                'top_companies': list(set([r['company'] for r in results[:3]])),
                'avg_score': sum([r['score'] for r in results]) / len(results) if results else 0
            })
        except Exception as e:
            report['sample_searches'].append({
                'query': query,
                'error': str(e)
            })
    
    # Add recommendations
    report['recommendations'] = [
        "Use semantic search for finding conceptually similar content across different companies",
        "Apply company-specific filters when looking for comparative analysis",
        "Combine multiple search queries for comprehensive topic coverage",
        "Monitor search performance and retrain embeddings if document types change significantly"
    ]
    
    return report

# Main execution
if __name__ == "__main__":
    
    print("Financial Vector Store Setup")
    print("=" * 40)
    
    # Test if required packages are available
    missing_packages = []
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing_packages.append("sentence-transformers")
    if not FAISS_AVAILABLE:
        missing_packages.append("faiss-cpu")
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        exit(1)
    
    # Run the test
    success = test_vector_store_functionality()
    
    if success:
        print("\n✅ Vector store setup completed successfully!")
        print("\nGenerated files:")
        print("  - data/embeddings/financial_index.faiss (FAISS index)")
        print("  - data/embeddings/metadata.db (SQLite metadata)")
        print("  - data/embeddings/embedded_chunks_*.pkl (serialized chunks)")
        
        print("\nNext steps:")
        print("  1. Use the vector store in your RAG pipeline")
        print("  2. Experiment with different embedding models")
        print("  3. Fine-tune retrieval parameters for your use case")
        print("  4. Add more sophisticated filtering and ranking")
    else:
        print("\n❌ Vector store setup failed. Check error messages above.")