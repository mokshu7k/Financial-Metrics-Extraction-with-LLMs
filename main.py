# main.py
"""
Main entry point for Financial RAG Research Pipeline.

This script orchestrates the complete workflow:
1. Data preprocessing (text cleaning, chunking)
2. Embedding generation and vector store creation
3. Extraction using three methods (Zero-Shot, CoT, RAG)
4. Comprehensive evaluation and comparison
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Import from src modules
from src.utils.logger import setup_logging, get_logger, ExperimentLogger
from src.utils.config_loader import ConfigLoader
from src.utils.helpers import timer, safe_json_save, ensure_directory, get_timestamp

from src.data_preprocessing.text_cleaning import FinancialTextCleaner
from src.data_preprocessing.chunking import FinancialDocumentChunker, ChunkType
from src.data_preprocessing.embed_store import (
    FinancialVectorStoreManager,
    create_financial_vector_store
)

from src.methods.zero_shot import ZeroShotExtractor
from src.methods.chain_of_thought import ChainOfThoughtExtractor
from src.methods.rag import RAGExtractor

from src.evaluation.evaluator import UnifiedEvaluator

# Import API clients (you'll need to install these)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class FinancialRAGPipeline:
    """
    Main pipeline orchestrator for the Financial RAG research project.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config = ConfigLoader(config_path)
        
        # Setup logging
        log_file = self.config.get('logging.file', 'results/logs/pipeline.log')
        log_level = self.config.get('logging.level', 'INFO')
        setup_logging(level=log_level, log_file=log_file, console=True)
        self.logger = get_logger(__name__)
        
        # Setup experiment logging
        self.exp_logger = ExperimentLogger(
            "financial_rag_pipeline",
            log_dir=str(Path(self.config.get('paths.results', 'results')) / "logs")
        )
        
        # Initialize components (lazy initialization)
        self.text_cleaner = None
        self.chunker = None
        self.vector_manager = None
        self.extractors = {}
        self.evaluator = None
        
        # API clients
        self.openai_client = None
        self.anthropic_client = None
        self.groq_client = None
        
        self.logger.info("Initialized Financial RAG Pipeline")
    
    def setup_api_clients(self):
        """Setup API clients for LLM access."""
        # # OpenAI
        # if OPENAI_AVAILABLE:
        #     import os
        #     api_key = os.getenv('OPENAI_API_KEY')
        #     if api_key:
        #         self.openai_client = openai.OpenAI(api_key=api_key)
        #         self.logger.info("OpenAI client initialized")
        #     else:
        #         self.logger.warning("OPENAI_API_KEY not found in environment")
        
        # # Anthropic
        # if ANTHROPIC_AVAILABLE:
        #     import os
        #     api_key = os.getenv('ANTHROPIC_API_KEY')
        #     if api_key:
        #         self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        #         self.logger.info("Anthropic client initialized")
        #     else:
        #         self.logger.warning("ANTHROPIC_API_KEY not found in environment")

        # Groq
        if GROQ_AVAILABLE:
            import os
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = groq.Groq(api_key=api_key)
                self.logger.info("Groq client initialized")
            else:
                self.logger.warning("GROQ_API_KEY not found in environment")

    
    @timer("Data preprocessing")
    def run_preprocessing(self, force: bool = False) -> Dict[str, Any]:
        """
        Run data preprocessing pipeline.
        
        Args:
            force: Force reprocessing even if files exist
        
        Returns:
            Preprocessing results
        """
        self.logger.info("="*80)
        self.logger.info("STEP 1: DATA PREPROCESSING")
        self.logger.info("="*80)
        
        processed_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        
        # Check if already processed
        if not force and (processed_dir / "financial_metadata.csv").exists():
            self.logger.info("Preprocessed files found. Skipping preprocessing.")
            self.logger.info("Use --force-preprocess to rerun.")
            return {'status': 'skipped', 'reason': 'already_exists'}
        
        # Initialize text cleaner
        self.text_cleaner = FinancialTextCleaner(config_path=str(self.config.config_path))
        
        # Run preprocessing
        results = self.text_cleaner.run_preprocessing_pipeline()
        
        if results:
            report = results.get('processing_report', {})
            self.exp_logger.log_metrics({
                'preprocessing_total_docs': report.get('summary', {}).get('total_documents', 0),
                'preprocessing_financial': report.get('summary', {}).get('financial_documents', 0),
                'preprocessing_transcripts': report.get('summary', {}).get('transcript_documents', 0)
            })
        
        return results
    
    @timer("Document chunking")
    def run_chunking(self, strategy: str = None, force: bool = False) -> Dict[str, Any]:
        """
        Run document chunking pipeline.
        
        Args:
            strategy: Chunking strategy (fixed_size, paragraph, section)
            force: Force re-chunking even if chunks exist
        
        Returns:
            Chunking results
        """
        self.logger.info("="*80)
        self.logger.info("STEP 2: DOCUMENT CHUNKING")
        self.logger.info("="*80)
        
        if strategy is None:
            strategy = self.config.get('preprocessing.chunking.default_strategy', 'fixed_size')
        
        chunks_dir = Path(self.config.get('paths.processed_data', 'data/processed')) / "chunks"
        
        # Check if already chunked
        if not force and (chunks_dir / f"financial_chunks_{strategy}.json").exists():
            self.logger.info(f"Chunks for strategy '{strategy}' found. Skipping chunking.")
            self.logger.info("Use --force-rechunk to rerun.")
            return {'status': 'skipped', 'reason': 'already_exists', 'strategy': strategy}
        
        # Initialize chunker
        self.chunker = FinancialDocumentChunker(config_path=str(self.config.config_path))
        
        # Map strategy string to enum
        strategy_map = {
            'fixed_size': ChunkType.FIXED_SIZE,
            'paragraph': ChunkType.PARAGRAPH,
            'section': ChunkType.SECTION
        }
        
        chunk_type = strategy_map.get(strategy, ChunkType.FIXED_SIZE)
        
        # Run chunking
        chunks = self.chunker.chunk_all_documents(chunk_type)
        self.chunker.save_chunks(chunks, output_suffix=f"_{strategy}")
        
        total_chunks = sum(len(chunk_list) for chunk_list in chunks.values())
        
        self.exp_logger.log_metrics({
            'chunking_strategy': strategy,
            'chunking_total_chunks': total_chunks,
            'chunking_financial': len(chunks['financial']),
            'chunking_transcript': len(chunks['transcript'])
        })
        
        return {
            'status': 'completed',
            'strategy': strategy,
            'total_chunks': total_chunks
        }
    
    @timer("Embedding generation")
    def run_embedding_generation(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate embeddings and create vector store.
        
        Args:
            force: Force re-embedding even if exists
        
        Returns:
            Embedding generation results
        """
        self.logger.info("="*80)
        self.logger.info("STEP 3: EMBEDDING GENERATION & VECTOR STORE")
        self.logger.info("="*80)
        
        embeddings_dir = Path(self.config.get('paths.embeddings', 'data/embeddings'))
        index_file = embeddings_dir / "financial_index.faiss"
        
        # Check if already exists
        if not force and index_file.exists():
            self.logger.info("Vector store found. Skipping embedding generation.")
            self.logger.info("Use --force-reembed to rerun.")
            
            # Load existing vector store
            self.vector_manager = create_financial_vector_store(
                config_path=str(self.config.config_path)
            )
            self.vector_manager.vector_store.load_index()
            
            return {'status': 'loaded', 'reason': 'already_exists'}
        
        # Create vector store manager
        self.vector_manager = create_financial_vector_store(
            config_path=str(self.config.config_path)
        )
        
        # Generate embeddings and create vector store
        strategy = self.config.get('preprocessing.chunking.default_strategy', 'fixed_size')
        num_chunks = self.vector_manager.create_vector_store_from_chunks(strategy)
        
        self.exp_logger.log_metrics({
            'embedding_model': self.vector_manager.embedding_config.model_name,
            'embedding_dimension': self.vector_manager.embedding_config.dimension,
            'embedding_num_chunks': num_chunks
        })
        
        return {
            'status': 'completed',
            'num_chunks': num_chunks,
            'model': self.vector_manager.embedding_config.model_name
        }
    
    @timer("Extraction with Zero-Shot")
    def run_zero_shot_extraction(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run zero-shot extraction on documents."""
        self.logger.info("="*80)
        self.logger.info("STEP 4a: ZERO-SHOT EXTRACTION")
        self.logger.info("="*80)
        
        # Get model configuration
        zero_shot_config = self.config.get('rag_methods.zero_shot', {})
        # Changed default model to Groq-supported Llama 3
        model_name = zero_shot_config.get('model', 'llama-3.3-70b-versatile')
        
        # Determine which client to use: Prioritize Groq
        if self.groq_client:
            api_client = self.groq_client
        # Remove/comment out the OpenAI and Anthropic client checks
        # elif 'gpt' in model_name.lower() and self.openai_client:
        #     api_client = self.openai_client
        # elif 'claude' in model_name.lower() and self.anthropic_client:
        #     api_client = self.anthropic_client
        else:
            self.logger.error(f"Groq API client not available for model: {model_name}")
            return {'error': 'No API client available'}
        
        # Initialize extractor
        extractor = ZeroShotExtractor(
            model_name=model_name,
            temperature=zero_shot_config.get('temperature', 0.1),
            max_tokens=zero_shot_config.get('max_tokens', 1000),
            api_client=api_client
        )
        
        # Run extraction
        results = extractor.batch_extract(
            documents=documents,
            output_dir="results/outputs/zero_shot"
        )
        
        # Log statistics
        stats = extractor.get_statistics()
        self.exp_logger.log_metrics({
            'zero_shot_model': model_name,
            'zero_shot_total_cost': stats['total_cost_usd'],
            'zero_shot_total_tokens': stats['total_tokens'],
            'zero_shot_num_requests': stats['total_requests']
        })
        
        return {
            'status': 'completed',
            'num_documents': len(results),
            'statistics': stats
        }
    
    @timer("Extraction with Chain-of-Thought")
    def run_cot_extraction(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run chain-of-thought extraction on documents."""
        self.logger.info("="*80)
        self.logger.info("STEP 4b: CHAIN-OF-THOUGHT EXTRACTION")
        self.logger.info("="*80)
        
        # Get model configuration
        cot_config = self.config.get('rag_methods.chain_of_thought', {})
        # Changed default model to Groq-supported Mixtral (a more capable model, like gpt-4)
        model_name = cot_config.get('model', 'llama-3.3-70b-versatile')
        
        # Determine which client to use: Prioritize Groq
        if self.groq_client:
            api_client = self.groq_client
        # Remove/comment out the OpenAI and Anthropic client checks
        # elif 'gpt' in model_name.lower() and self.openai_client:
        #     api_client = self.openai_client
        # elif 'claude' in model_name.lower() and self.anthropic_client:
        #     api_client = self.anthropic_client
        else:
            self.logger.error(f"Groq API client not available for model: {model_name}")
            return {'error': 'No API client available'}
        
        # Initialize extractor
        extractor = ChainOfThoughtExtractor(
            model_name=model_name,
            temperature=cot_config.get('temperature', 0.1),
            max_tokens=cot_config.get('max_tokens', 2000),
            api_client=api_client
        )
        
        # Run extraction
        results = extractor.batch_extract(
            documents=documents,
            output_dir="results/outputs/chain_of_thought"
        )
        
        # Log statistics
        stats = extractor.get_statistics()
        self.exp_logger.log_metrics({
            'cot_model': model_name,
            'cot_total_cost': stats['total_cost_usd'],
            'cot_total_tokens': stats['total_tokens'],
            'cot_num_requests': stats['total_requests']
        })
        
        return {
            'status': 'completed',
            'num_documents': len(results),
            'statistics': stats
        }
    
    @timer("Extraction with RAG")
    def run_rag_extraction(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run RAG-based extraction on documents."""
        self.logger.info("="*80)
        self.logger.info("STEP 4c: RAG EXTRACTION")
        self.logger.info("="*80)
        
        # Ensure vector store is loaded
        if not self.vector_manager:
            self.logger.error("Vector store not initialized. Run embedding generation first.")
            return {'error': 'Vector store not available'}
        
        # Get model configuration
        rag_config = self.config.get('rag_methods.rag', {})
        # Changed default model to Groq-supported Mixtral
        model_name = rag_config.get('model', 'llama-3.3-70b-versatile')
        
        # Determine which client to use: Prioritize Groq
        if self.groq_client:
            api_client = self.groq_client
        # Remove/comment out the OpenAI and Anthropic client checks
        # elif 'gpt' in model_name.lower() and self.openai_client:
        #     api_client = self.openai_client
        # elif 'claude' in model_name.lower() and self.anthropic_client:
        #     api_client = self.anthropic_client
        else:
            self.logger.error(f"Groq API client not available for model: {model_name}")
            return {'error': 'No API client available'}
        
        # Initialize extractor
        extractor = RAGExtractor(
            model_name=model_name,
            temperature=rag_config.get('temperature', 0.1),
            max_tokens=rag_config.get('max_tokens', 2000),
            retrieval_k=rag_config.get('retrieval.semantic_search', 5) if isinstance(rag_config.get('retrieval'), dict) else 5,
            vector_store_manager=self.vector_manager,
            api_client=api_client
        )
        
        # Run extraction
        results = extractor.batch_extract(
            documents=documents,
            output_dir="results/outputs/rag"
        )
        
        # Log statistics
        stats = extractor.get_statistics()
        self.exp_logger.log_metrics({
            'rag_model': model_name,
            'rag_total_cost': stats['total_cost_usd'],
            'rag_total_tokens': stats['total_tokens'],
            'rag_num_requests': stats['total_requests'],
            'rag_num_retrievals': stats['total_retrievals']
        })
        
        return {
            'status': 'completed',
            'num_documents': len(results),
            'statistics': stats
        }
    
    def load_documents_for_extraction(self) -> List[Dict[str, Any]]:
        """Load processed documents for extraction."""
        documents = []
        processed_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        
        # Load financial documents
        financial_dir = processed_dir / "financial"
        if financial_dir.exists():
            for file_path in financial_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    documents.append(doc)
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
        
        # Load transcript documents
        transcript_dir = processed_dir / "transcript"
        if transcript_dir.exists():
            for file_path in transcript_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    documents.append(doc)
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(documents)} documents for extraction")
        return documents
    
    @timer("Evaluation")
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of all methods."""
        self.logger.info("="*80)
        self.logger.info("STEP 5: EVALUATION & COMPARISON")
        self.logger.info("="*80)
        
        # Initialize evaluator
        self.evaluator = UnifiedEvaluator(config_path=str(self.config.config_path))
        
        # Run evaluation
        comparison_results = self.evaluator.evaluate_all_methods()
        
        # Print summary
        self.evaluator.print_summary(comparison_results)
        
        # Generate reports
        json_report = self.evaluator.generate_evaluation_report(
            comparison_results, output_format='json'
        )
        csv_report = self.evaluator.generate_evaluation_report(
            comparison_results, output_format='csv'
        )
        html_report = self.evaluator.generate_evaluation_report(
            comparison_results, output_format='html'
        )
        
        self.logger.info(f"\nEvaluation reports generated:")
        self.logger.info(f"  - JSON: {json_report}")
        self.logger.info(f"  - CSV: {csv_report}")
        self.logger.info(f"  - HTML: {html_report}")
        
        return comparison_results
    
    def run_full_pipeline(
        self,
        skip_preprocessing: bool = False,
        skip_chunking: bool = False,
        skip_embedding: bool = False,
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            skip_preprocessing: Skip preprocessing step
            skip_chunking: Skip chunking step
            skip_embedding: Skip embedding generation
            methods_to_run: List of methods to run (zero_shot, chain_of_thought, rag)
        
        Returns:
            Complete pipeline results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("FINANCIAL RAG RESEARCH PIPELINE")
        self.logger.info("="*80)
        
        # Setup API clients
        self.setup_api_clients()
        
        # Step 1: Preprocessing
        if not skip_preprocessing:
            preprocessing_results = self.run_preprocessing()
        else:
            self.logger.info("Skipping preprocessing (--skip-preprocess)")
            preprocessing_results = {'status': 'skipped'}
        
        # Step 2: Chunking
        if not skip_chunking:
            chunking_results = self.run_chunking()
        else:
            self.logger.info("Skipping chunking (--skip-chunk)")
            chunking_results = {'status': 'skipped'}
        
        # Step 3: Embedding Generation
        if not skip_embedding:
            embedding_results = self.run_embedding_generation()
        else:
            self.logger.info("Skipping embedding generation (--skip-embed)")
            embedding_results = {'status': 'skipped'}
        
        # Load documents for extraction
        documents = self.load_documents_for_extraction()
        
        if not documents:
            self.logger.error("No documents available for extraction!")
            return {'error': 'No documents found'}
        
        # Step 4: Run Extraction Methods
        if methods_to_run is None:
            methods_to_run = ['zero_shot', 'chain_of_thought', 'rag']
        
        extraction_results = {}
        
        if 'zero_shot' in methods_to_run:
            try:
                extraction_results['zero_shot'] = self.run_zero_shot_extraction(documents)
            except Exception as e:
                self.logger.error(f"Zero-shot extraction failed: {e}")
                extraction_results['zero_shot'] = {'error': str(e)}
        
        if 'chain_of_thought' in methods_to_run:
            try:
                extraction_results['chain_of_thought'] = self.run_cot_extraction(documents)
            except Exception as e:
                self.logger.error(f"CoT extraction failed: {e}")
                extraction_results['chain_of_thought'] = {'error': str(e)}
        
        if 'rag' in methods_to_run:
            try:
                extraction_results['rag'] = self.run_rag_extraction(documents)
            except Exception as e:
                self.logger.error(f"RAG extraction failed: {e}")
                extraction_results['rag'] = {'error': str(e)}
        
        # Step 5: Evaluation
        evaluation_results = self.run_evaluation()
        
        # Final summary
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)
        
        return {
            'preprocessing': preprocessing_results,
            'chunking': chunking_results,
            'embedding': embedding_results,
            'extraction': extraction_results,
            'evaluation': evaluation_results
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial RAG Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --run-all
  
  # Run only preprocessing
  python main.py --preprocess
  
  # Run extraction with specific methods
  python main.py --extract --methods zero_shot rag
  
  # Run evaluation only
  python main.py --evaluate
  
  # Force reprocessing
  python main.py --run-all --force-preprocess --force-rechunk
        """
    )
    
    # Pipeline stages
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing only')
    parser.add_argument('--chunk', action='store_true',
                       help='Run chunking only')
    parser.add_argument('--embed', action='store_true',
                       help='Run embedding generation only')
    parser.add_argument('--extract', action='store_true',
                       help='Run extraction only')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only')
    
    # Options
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--methods', nargs='+',
                       choices=['zero_shot', 'chain_of_thought', 'rag'],
                       help='Methods to run for extraction')
    parser.add_argument('--chunk-strategy', type=str,
                       choices=['fixed_size', 'paragraph', 'section'],
                       help='Chunking strategy to use')
    
    # Force options
    parser.add_argument('--force-preprocess', action='store_true',
                       help='Force reprocessing even if files exist')
    parser.add_argument('--force-rechunk', action='store_true',
                       help='Force re-chunking even if chunks exist')
    parser.add_argument('--force-reembed', action='store_true',
                       help='Force re-embedding even if vector store exists')
    
    # Skip options
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-chunk', action='store_true',
                       help='Skip chunking step')
    parser.add_argument('--skip-embed', action='store_true',
                       help='Skip embedding generation step')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Initialize pipeline
    pipeline = FinancialRAGPipeline(config_path=args.config)
    
    try:
        if args.run_all:
            # Run complete pipeline
            results = pipeline.run_full_pipeline(
                skip_preprocessing=args.skip_preprocess,
                skip_chunking=args.skip_chunk,
                skip_embedding=args.skip_embed,
                methods_to_run=args.methods
            )
            
        elif args.preprocess:
            # Run preprocessing only
            pipeline.run_preprocessing(force=args.force_preprocess)
            
        elif args.chunk:
            # Run chunking only
            pipeline.run_chunking(
                strategy=args.chunk_strategy,
                force=args.force_rechunk
            )
            
        elif args.embed:
            # Run embedding generation only
            pipeline.run_embedding_generation(force=args.force_reembed)
            
        elif args.extract:
            # Run extraction only
            pipeline.setup_api_clients()
            documents = pipeline.load_documents_for_extraction()
            
            if not documents:
                pipeline.logger.error("No documents found for extraction")
                sys.exit(1)
            
            methods = args.methods or ['zero_shot', 'chain_of_thought', 'rag']
            
            if 'zero_shot' in methods:
                pipeline.run_zero_shot_extraction(documents)
            if 'chain_of_thought' in methods:
                pipeline.run_cot_extraction(documents)
            if 'rag' in methods:
                pipeline.run_rag_extraction(documents)
                
        elif args.evaluate:
            # Run evaluation only
            pipeline.run_evaluation()
            
        else:
            print("No action specified. Use --help for usage information.")
            sys.exit(1)
        
        print("\n" + "="*80)
        print("SUCCESS: Pipeline completed without errors")
        print("="*80)
        
    except KeyboardInterrupt:
        pipeline.logger.warning("\nPipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        pipeline.logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()