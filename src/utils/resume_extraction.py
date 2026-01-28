"""
Utility to resume extraction from where it left off.
Handles resumption for Zero-Shot, Chain-of-Thought, and RAG methods.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class ExtractionResume:
    """Helper class to manage extraction resumption."""
    
    def __init__(self, output_dir: str):
        """
        Initialize resume manager.
        
        Args:
            output_dir: Directory where extraction results are saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_processed_documents(self) -> Set[str]:
        """
        Get set of already processed document identifiers.
        
        Returns:
            Set of document identifiers (format: "COMPANY_QUARTER_YEAR")
        """
        processed = set()
        
        for file_path in self.output_dir.glob("*.json"):
            # Skip summary files
            if 'summary' in file_path.name.lower():
                continue
            
            try:
                # Try to extract document ID from filename
                # Format: COMPANY_QUARTER_YEAR.json
                stem = file_path.stem
                processed.add(stem)
                
                # Also try to read the file and get metadata
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract document ID from content
                extraction = data.get('extraction', {})
                if extraction and not extraction.get('error'):
                    company = extraction.get('company', '')
                    quarter = extraction.get('quarter', '')
                    year = extraction.get('year', '')
                    
                    if company and quarter and year:
                        doc_id = f"{company}_{quarter}_{year}"
                        processed.add(doc_id)
                        
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        return processed
    
    def filter_unprocessed_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out already processed documents.
        
        Args:
            documents: List of all documents
            
        Returns:
            List of unprocessed documents only
        """
        processed = self.get_processed_documents()
        
        unprocessed = []
        for doc in documents:
            # Extract document metadata
            metadata = doc.get('metadata', {})
            company = metadata.get('company', doc.get('company', 'Unknown'))
            quarter = metadata.get('quarter', doc.get('quarter', 'Unknown'))
            year = metadata.get('year', doc.get('year', 0))
            
            doc_id = f"{company}_{quarter}_{year}"
            
            if doc_id not in processed:
                unprocessed.append(doc)
            else:
                logger.info(f"Skipping already processed: {doc_id}")
        
        logger.info(
            f"Found {len(processed)} processed documents, "
            f"{len(unprocessed)} remaining to process"
        )
        
        return unprocessed
    
    def get_resume_stats(self, total_documents: int) -> Dict[str, Any]:
        """
        Get statistics about resume state.
        
        Args:
            total_documents: Total number of documents to process
            
        Returns:
            Dictionary with resume statistics
        """
        processed = self.get_processed_documents()
        
        return {
            'total_documents': total_documents,
            'processed_documents': len(processed),
            'remaining_documents': total_documents - len(processed),
            'progress_percentage': (len(processed) / total_documents * 100) if total_documents > 0 else 0,
            'processed_ids': sorted(list(processed))
        }


def resume_zero_shot_extraction(
    documents: List[Dict[str, Any]],
    output_dir: str = "results/outputs/zero_shot",
    **extractor_kwargs
):
    """
    Resume zero-shot extraction from where it left off.
    
    Args:
        documents: All documents to process
        output_dir: Output directory
        **extractor_kwargs: Arguments to pass to ZeroShotExtractor
        
    Returns:
        Extraction results for newly processed documents
    """
    from src.methods.zero_shot import ZeroShotExtractor
    
    # Initialize resume manager
    resume_manager = ExtractionResume(output_dir)
    
    # Get resume stats
    stats = resume_manager.get_resume_stats(len(documents))
    logger.info(f"Zero-Shot Resume Stats: {stats['processed_documents']}/{stats['total_documents']} completed")
    
    # Filter unprocessed documents
    unprocessed_docs = resume_manager.filter_unprocessed_documents(documents)
    
    if not unprocessed_docs:
        logger.info("✓ All documents already processed for Zero-Shot!")
        return []
    
    logger.info(f"Resuming Zero-Shot extraction for {len(unprocessed_docs)} documents...")
    
    # Initialize extractor
    extractor = ZeroShotExtractor(**extractor_kwargs)
    
    # Process remaining documents
    results = extractor.batch_extract(
        documents=unprocessed_docs,
        output_dir=output_dir
    )
    
    # Update final stats
    final_stats = resume_manager.get_resume_stats(len(documents))
    logger.info(
        f"Zero-Shot extraction complete: {final_stats['processed_documents']}/{final_stats['total_documents']} "
        f"({final_stats['progress_percentage']:.1f}%)"
    )
    
    return results


def resume_cot_extraction(
    documents: List[Dict[str, Any]],
    output_dir: str = "results/outputs/chain_of_thought",
    **extractor_kwargs
):
    """
    Resume chain-of-thought extraction from where it left off.
    
    Args:
        documents: All documents to process
        output_dir: Output directory
        **extractor_kwargs: Arguments to pass to ChainOfThoughtExtractor
        
    Returns:
        Extraction results for newly processed documents
    """
    from src.methods.chain_of_thought import ChainOfThoughtExtractor
    
    # Initialize resume manager
    resume_manager = ExtractionResume(output_dir)
    
    # Get resume stats
    stats = resume_manager.get_resume_stats(len(documents))
    logger.info(f"CoT Resume Stats: {stats['processed_documents']}/{stats['total_documents']} completed")
    
    # Filter unprocessed documents
    unprocessed_docs = resume_manager.filter_unprocessed_documents(documents)
    
    if not unprocessed_docs:
        logger.info("✓ All documents already processed for Chain-of-Thought!")
        return []
    
    logger.info(f"Resuming Chain-of-Thought extraction for {len(unprocessed_docs)} documents...")
    
    # Initialize extractor
    extractor = ChainOfThoughtExtractor(**extractor_kwargs)
    
    # Process remaining documents
    results = extractor.batch_extract(
        documents=unprocessed_docs,
        output_dir=output_dir
    )
    
    # Update final stats
    final_stats = resume_manager.get_resume_stats(len(documents))
    logger.info(
        f"CoT extraction complete: {final_stats['processed_documents']}/{final_stats['total_documents']} "
        f"({final_stats['progress_percentage']:.1f}%)"
    )
    
    return results


def resume_rag_extraction(
    documents: List[Dict[str, Any]],
    vector_store_manager,
    output_dir: str = "results/outputs/rag",
    **extractor_kwargs
):
    """
    Resume RAG extraction from where it left off.
    
    Args:
        documents: All documents to process
        vector_store_manager: Vector store manager instance
        output_dir: Output directory
        **extractor_kwargs: Arguments to pass to RAGExtractor
        
    Returns:
        Extraction results for newly processed documents
    """
    from src.methods.rag import RAGExtractor
    
    # Initialize resume manager
    resume_manager = ExtractionResume(output_dir)
    
    # Get resume stats
    stats = resume_manager.get_resume_stats(len(documents))
    logger.info(f"RAG Resume Stats: {stats['processed_documents']}/{stats['total_documents']} completed")
    
    # Filter unprocessed documents
    unprocessed_docs = resume_manager.filter_unprocessed_documents(documents)
    
    if not unprocessed_docs:
        logger.info("✓ All documents already processed for RAG!")
        return []
    
    logger.info(f"Resuming RAG extraction for {len(unprocessed_docs)} documents...")
    
    # Initialize extractor
    extractor = RAGExtractor(
        vector_store_manager=vector_store_manager,
        **extractor_kwargs
    )
    
    # Process remaining documents
    results = extractor.batch_extract(
        documents=unprocessed_docs,
        output_dir=output_dir
    )
    
    # Update final stats
    final_stats = resume_manager.get_resume_stats(len(documents))
    logger.info(
        f"RAG extraction complete: {final_stats['processed_documents']}/{final_stats['total_documents']} "
        f"({final_stats['progress_percentage']:.1f}%)"
    )
    
    return results


def check_extraction_progress(method: str = "all"):
    """
    Check progress of extraction methods.
    
    Args:
        method: Method to check ('zero_shot', 'chain_of_thought', 'rag', or 'all')
    """
    methods_to_check = []
    
    if method == "all":
        methods_to_check = [
            ("zero_shot", "results/outputs/zero_shot"),
            ("chain_of_thought", "results/outputs/chain_of_thought"),
            ("rag", "results/outputs/rag")
        ]
    else:
        output_dir = f"results/outputs/{method}"
        methods_to_check = [(method, output_dir)]
    
    # Load all documents to get total count
    from pathlib import Path
    import json
    
    documents = []
    processed_dir = Path("data/processed")
    
    for doc_dir in [processed_dir / "financial", processed_dir / "transcript"]:
        if doc_dir.exists():
            for file_path in doc_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    documents.append(doc)
                except:
                    pass
    
    total_docs = len(documents)
    
    print("\n" + "="*80)
    print("EXTRACTION PROGRESS REPORT")
    print("="*80)
    print(f"Total documents to process: {total_docs}\n")
    
    for method_name, output_dir in methods_to_check:
        resume_manager = ExtractionResume(output_dir)
        stats = resume_manager.get_resume_stats(total_docs)
        
        print(f"{method_name.upper()}:")
        print(f"  Processed: {stats['processed_documents']}/{stats['total_documents']}")
        print(f"  Remaining: {stats['remaining_documents']}")
        print(f"  Progress: {stats['progress_percentage']:.1f}%")
        
        if stats['remaining_documents'] > 0:
            print(f"  Status: ⚠ INCOMPLETE - {stats['remaining_documents']} documents remaining")
        else:
            print(f"  Status: ✓ COMPLETE")
        print()
    
    print("="*80)


# Command-line interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Resume extraction where you left off")
    
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Check progress of all methods'
    )
    
    parser.add_argument(
        '--method',
        choices=['zero_shot', 'chain_of_thought', 'rag', 'all'],
        default='all',
        help='Method to check or resume'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume extraction for specified method'
    )
    
    args = parser.parse_args()
    
    if args.check:
        # Check progress
        check_extraction_progress(args.method)
    
    elif args.resume:
        print("To resume extraction, use main.py with --extract flag")
        print("The extraction will automatically skip already processed documents")
        sys.exit(0)
    
    else:
        parser.print_help()