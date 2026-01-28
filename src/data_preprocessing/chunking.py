import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum

# Import from utils
from ..utils.logger import setup_logging, get_logger
from ..utils.config_loader import ConfigLoader
from ..utils.helpers import timer, safe_json_save, ensure_directory, get_timestamp

class ChunkType(Enum):
    """Types of chunks for different processing strategies."""
    SEMANTIC = "semantic"          # Topic-based chunks
    FIXED_SIZE = "fixed_size"      # Fixed token/character count
    PARAGRAPH = "paragraph"        # Natural paragraph breaks
    SECTION = "section"            # Document sections (for transcripts)

@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    chunk_id: str
    document_id: str
    company: str
    quarter: str
    year: int
    document_type: str
    chunk_type: str
    chunk_index: int
    start_position: int
    end_position: int
    token_count: int
    contains_financial_data: bool
    section_type: Optional[str] = None

@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    metadata: ChunkMetadata

class FinancialDocumentChunker:
    """
    Chunks financial documents for optimal RAG performance.
    Handles both financial filings and earnings call transcripts.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = ConfigLoader(config_path)
        
        # Setup paths
        self.processed_data_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        
        # Get chunking parameters from config
        chunking_config = self.config.get('preprocessing.chunking', {})
        self.chunk_size = chunking_config.get('chunk_size', 1000)
        self.chunk_overlap = chunking_config.get('chunk_overlap', 200)
        self.min_chunk_size = chunking_config.get('min_chunk_size', 100)
        
        # Setup logging
        log_file = self.config.get('logging.file', 'data/logs/pipeline.log')
        log_level = self.config.get('logging.level', 'INFO')
        setup_logging(level=log_level, log_file=log_file)
        self.logger = get_logger(__name__)
        
        # Patterns for identifying financial data
        self.financial_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:crore|million|billion|thousand)',
            r'revenue|profit|loss|expenses|ebitda|eps|margin',
            r'growth|increase|decrease|decline',
            r'quarter|q[1-4]|fy\s*\d{2,4}|year'
        ]
        
        # Patterns for section identification in transcripts
        self.section_patterns = {
            'management_discussion': [
                r'management\s+discuss',
                r'prepared\s+remarks',
                r'opening\s+statement',
                r'varun\s+berry|managing\s+director|ceo'
            ],
            'qa_session': [
                r'question.*answer',
                r'q\s*&\s*a',
                r'moderator.*question',
                r'first\s+question'
            ],
            'financial_metrics': [
                r'financial\s+results',
                r'key\s+metrics',
                r'performance\s+highlights'
            ]
        }
    
    def estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def contains_financial_data(self, text: str) -> bool:
        """Check if text contains financial data/metrics."""
        text_lower = text.lower()
        for pattern in self.financial_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def identify_section_type(self, text: str) -> Optional[str]:
        """Identify the type of section in transcript text."""
        text_lower = text.lower()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section_type
        
        return None
    
    @timer("Fixed-size chunking")
    def chunk_by_fixed_size(self, text: str, metadata_base: Dict) -> List[DocumentChunk]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundaries
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.min_chunk_size:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = f"{metadata_base['document_id']}_chunk_{chunk_index}"
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=metadata_base['document_id'],
                    company=metadata_base['company'],
                    quarter=metadata_base['quarter'],
                    year=metadata_base['year'],
                    document_type=metadata_base['document_type'],
                    chunk_type=ChunkType.FIXED_SIZE.value,
                    chunk_index=chunk_index,
                    start_position=start,
                    end_position=end,
                    token_count=self.estimate_token_count(chunk_text),
                    contains_financial_data=self.contains_financial_data(chunk_text),
                    section_type=self.identify_section_type(chunk_text)
                )
                
                chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
            # Prevent infinite loops
            if start >= len(text):
                break
        
        return chunks
    
    @timer("Paragraph-based chunking")
    def chunk_by_paragraphs(self, text: str, metadata_base: Dict) -> List[DocumentChunk]:
        """Create chunks based on paragraph boundaries."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph exceeds chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.estimate_token_count(potential_chunk) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_id = f"{metadata_base['document_id']}_para_{chunk_index}"
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=metadata_base['document_id'],
                    company=metadata_base['company'],
                    quarter=metadata_base['quarter'],
                    year=metadata_base['year'],
                    document_type=metadata_base['document_type'],
                    chunk_type=ChunkType.PARAGRAPH.value,
                    chunk_index=chunk_index,
                    start_position=start_position,
                    end_position=start_position + len(current_chunk),
                    token_count=self.estimate_token_count(current_chunk),
                    contains_financial_data=self.contains_financial_data(current_chunk),
                    section_type=self.identify_section_type(current_chunk)
                )
                
                chunks.append(DocumentChunk(content=current_chunk, metadata=metadata))
                
                # Start new chunk
                current_chunk = paragraph
                start_position += len(current_chunk)
                chunk_index += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_id = f"{metadata_base['document_id']}_para_{chunk_index}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=metadata_base['document_id'],
                company=metadata_base['company'],
                quarter=metadata_base['quarter'],
                year=metadata_base['year'],
                document_type=metadata_base['document_type'],
                chunk_type=ChunkType.PARAGRAPH.value,
                chunk_index=chunk_index,
                start_position=start_position,
                end_position=start_position + len(current_chunk),
                token_count=self.estimate_token_count(current_chunk),
                contains_financial_data=self.contains_financial_data(current_chunk),
                section_type=self.identify_section_type(current_chunk)
            )
            
            chunks.append(DocumentChunk(content=current_chunk, metadata=metadata))
        
        return chunks
    
    @timer("Section-based chunking")
    def chunk_transcript_by_sections(self, document: Dict, metadata_base: Dict) -> List[DocumentChunk]:
        """Chunk transcript by identified sections (management discussion, Q&A, etc.)."""
        sections = document.get('sections', {})
        chunks = []
        chunk_index = 0
        
        for section_name, section_text in sections.items():
            if not section_text or len(section_text) < self.min_chunk_size:
                continue
            
            # If section is too large, further chunk it
            if self.estimate_token_count(section_text) > self.chunk_size:
                sub_chunks = self.chunk_by_fixed_size(section_text, metadata_base)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.section_type = section_name
                    sub_chunk.metadata.chunk_type = ChunkType.SECTION.value
                    chunks.append(sub_chunk)
            else:
                # Use entire section as one chunk
                chunk_id = f"{metadata_base['document_id']}_section_{section_name}"
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=metadata_base['document_id'],
                    company=metadata_base['company'],
                    quarter=metadata_base['quarter'],
                    year=metadata_base['year'],
                    document_type=metadata_base['document_type'],
                    chunk_type=ChunkType.SECTION.value,
                    chunk_index=chunk_index,
                    start_position=0,
                    end_position=len(section_text),
                    token_count=self.estimate_token_count(section_text),
                    contains_financial_data=self.contains_financial_data(section_text),
                    section_type=section_name
                )
                
                chunks.append(DocumentChunk(content=section_text, metadata=metadata))
                chunk_index += 1
        
        return chunks
    
    def chunk_single_document(self, document: Dict, chunk_type: ChunkType = ChunkType.FIXED_SIZE) -> List[DocumentChunk]:
        """Chunk a single document using the specified strategy."""
        metadata = document['metadata']
        
        # Create base metadata for chunks
        metadata_base = {
            'document_id': f"{metadata['company']}_{metadata['quarter']}_{metadata['year']}_{metadata['document_type']}",
            'company': metadata['company'],
            'quarter': metadata['quarter'],
            'year': metadata['year'],
            'document_type': metadata['document_type']
        }
        
        # Choose chunking strategy based on document type and chunk type
        if chunk_type == ChunkType.SECTION and metadata['document_type'] == 'transcript':
            return self.chunk_transcript_by_sections(document, metadata_base)
        elif chunk_type == ChunkType.PARAGRAPH:
            return self.chunk_by_paragraphs(document['cleaned_text'], metadata_base)
        else:  # Default to fixed size
            return self.chunk_by_fixed_size(document['cleaned_text'], metadata_base)
    
    @timer("Chunking all documents")
    def chunk_all_documents(self, chunk_type: ChunkType = ChunkType.FIXED_SIZE) -> Dict[str, List[DocumentChunk]]:
        """Chunk all processed documents."""
        all_chunks = {
            'financial': [],
            'transcript': []
        }
        
        # Process financial documents
        financial_dir = self.processed_data_dir / "financial"
        if financial_dir.exists():
            for file_path in financial_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    chunks = self.chunk_single_document(document, chunk_type)
                    all_chunks['financial'].extend(chunks)
                    
                    self.logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    self.logger.error(f"Error chunking financial document {file_path.name}: {e}")
        
        # Process transcript documents
        transcript_dir = self.processed_data_dir / "transcript"
        if transcript_dir.exists():
            for file_path in transcript_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    chunks = self.chunk_single_document(document, chunk_type)
                    all_chunks['transcript'].extend(chunks)
                    
                    self.logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    self.logger.error(f"Error chunking transcript document {file_path.name}: {e}")
        
        return all_chunks
    
    def save_chunks(self, chunks: Dict[str, List[DocumentChunk]], output_suffix: str = ""):
        """Save chunks to JSON files for embedding and retrieval."""
        chunks_dir = ensure_directory(self.processed_data_dir / "chunks")
        
        all_chunk_metadata = []
        
        for doc_type, chunk_list in chunks.items():
            if not chunk_list:
                continue
            
            # Prepare data for saving
            chunks_data = []
            for chunk in chunk_list:
                chunk_dict = {
                    'content': chunk.content,
                    'metadata': chunk.metadata.__dict__
                }
                chunks_data.append(chunk_dict)
                all_chunk_metadata.append(chunk.metadata.__dict__)
            
            # Save chunks
            filename = f"{doc_type}_chunks{output_suffix}.json"
            safe_json_save(chunks_data, chunks_dir / filename)
            
            self.logger.info(f"Saved {len(chunks_data)} {doc_type} chunks to {filename}")
        
        # Save combined metadata
        if all_chunk_metadata:
            metadata_df = pd.DataFrame(all_chunk_metadata)
            metadata_df.to_csv(chunks_dir / f"chunks_metadata{output_suffix}.csv", index=False)
            
            # Generate chunking statistics
            stats = self.generate_chunking_statistics(metadata_df)
            safe_json_save(stats, chunks_dir / f"chunking_stats{output_suffix}.json")
    
    def generate_chunking_statistics(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics about the chunking process."""
        stats = {
            'total_chunks': len(metadata_df),
            'chunk_types': metadata_df['chunk_type'].value_counts().to_dict(),
            'document_types': metadata_df['document_type'].value_counts().to_dict(),
            'companies': metadata_df['company'].value_counts().to_dict(),
            'years': sorted(metadata_df['year'].unique().tolist()),
            'token_statistics': {
                'mean': float(metadata_df['token_count'].mean()),
                'median': float(metadata_df['token_count'].median()),
                'min': int(metadata_df['token_count'].min()),
                'max': int(metadata_df['token_count'].max()),
                'std': float(metadata_df['token_count'].std())
            },
            'financial_data_chunks': int(metadata_df['contains_financial_data'].sum()),
            'section_types': metadata_df['section_type'].value_counts().to_dict() if 'section_type' in metadata_df.columns else {}
        }
        
        return stats
    
    @timer("Chunking strategy comparison")
    def create_chunking_strategy_comparison(self) -> Dict[str, Any]:
        """Compare different chunking strategies on the same documents."""
        comparison_results = {}
        
        for chunk_type in [ChunkType.FIXED_SIZE, ChunkType.PARAGRAPH, ChunkType.SECTION]:
            self.logger.info(f"Testing chunking strategy: {chunk_type.value}")
            
            chunks = self.chunk_all_documents(chunk_type)
            self.save_chunks(chunks, output_suffix=f"_{chunk_type.value}")
            
            # Calculate metrics for this strategy
            total_chunks = sum(len(chunk_list) for chunk_list in chunks.values())
            
            if total_chunks > 0:
                all_chunk_metadata = []
                for chunk_list in chunks.values():
                    all_chunk_metadata.extend([chunk.metadata.__dict__ for chunk in chunk_list])
                
                metadata_df = pd.DataFrame(all_chunk_metadata)
                
                comparison_results[chunk_type.value] = {
                    'total_chunks': total_chunks,
                    'avg_tokens_per_chunk': float(metadata_df['token_count'].mean()),
                    'financial_data_coverage': float(metadata_df['contains_financial_data'].mean()),
                    'chunk_size_variance': float(metadata_df['token_count'].std())
                }
        
        # Save comparison results
        safe_json_save(comparison_results, self.processed_data_dir / "chunking_comparison.json")
        
        return comparison_results
    
    def get_chunk_recommendations(self, document_type: str = None) -> Dict[str, str]:
        """Provide recommendations for optimal chunking strategy."""
        recommendations = {
            'general': "For most RAG applications, FIXED_SIZE with 800-1200 tokens provides good balance of context and retrieval precision.",
            
            'financial_filings': "Use PARAGRAPH chunking for financial filings to maintain natural document structure and keep related financial metrics together.",
            
            'transcripts': "Use SECTION chunking for earnings call transcripts to separate management discussion from Q&A, enabling targeted retrieval.",
            
            'small_documents': "For documents < 2000 tokens, consider using the entire document as a single chunk to avoid fragmentation.",
            
            'large_documents': "For documents > 10000 tokens, use FIXED_SIZE with moderate overlap (150-200 tokens) to ensure comprehensive coverage."
        }
        
        if document_type:
            if document_type == 'financial':
                return {'recommendation': recommendations['financial_filings']}
            elif document_type == 'transcript':
                return {'recommendation': recommendations['transcripts']}
        
        return recommendations


def analyze_chunk_quality(chunks_metadata_path: Path) -> Dict[str, Any]:
    """Analyze the quality of generated chunks."""
    metadata_df = pd.read_csv(chunks_metadata_path)
    
    analysis = {
        'size_distribution': {
            'under_100_tokens': int((metadata_df['token_count'] < 100).sum()),
            '100_500_tokens': int(((metadata_df['token_count'] >= 100) & (metadata_df['token_count'] < 500)).sum()),
            '500_1000_tokens': int(((metadata_df['token_count'] >= 500) & (metadata_df['token_count'] < 1000)).sum()),
            'over_1000_tokens': int((metadata_df['token_count'] >= 1000).sum())
        },
        'content_quality': {
            'chunks_with_financial_data': int(metadata_df['contains_financial_data'].sum()),
            'empty_or_short_chunks': int((metadata_df['token_count'] < 50).sum()),
            'very_long_chunks': int((metadata_df['token_count'] > 2000).sum())
        },
        'coverage': {
            'companies_covered': int(metadata_df['company'].nunique()),
            'years_covered': len(metadata_df['year'].unique()),
            'document_types': metadata_df['document_type'].value_counts().to_dict()
        }
    }
    
    return analysis


def optimize_chunk_parameters(sample_texts: List[str], target_chunk_size: int = 1000) -> Dict[str, int]:
    """Optimize chunking parameters based on sample texts."""
    # Analyze text characteristics
    avg_paragraph_length = np.mean([len(para) for text in sample_texts for para in text.split('\n\n')])
    avg_sentence_length = np.mean([len(sent) for text in sample_texts for sent in text.split('.')])
    
    # Calculate optimal parameters
    optimal_chunk_size = min(target_chunk_size, int(avg_paragraph_length * 3))
    optimal_overlap = min(200, int(optimal_chunk_size * 0.2))
    min_chunk_size = max(50, int(avg_sentence_length * 2))
    
    return {
        'chunk_size': optimal_chunk_size,
        'overlap': optimal_overlap,
        'min_chunk_size': min_chunk_size
    }


# Main execution
if __name__ == "__main__":
    chunker = FinancialDocumentChunker(config_path="config.yaml")
    
    print("Starting document chunking process...")
    
    # Test different chunking strategies
    comparison_results = chunker.create_chunking_strategy_comparison()
    
    print("\nChunking Strategy Comparison:")
    print("=" * 50)
    for strategy, metrics in comparison_results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Total chunks: {metrics['total_chunks']}")
        print(f"  Avg tokens per chunk: {metrics['avg_tokens_per_chunk']:.1f}")
        print(f"  Financial data coverage: {metrics['financial_data_coverage']:.1%}")
        print(f"  Size variance: {metrics['chunk_size_variance']:.1f}")
    
    # Get recommendations
    recommendations = chunker.get_chunk_recommendations()
    print(f"\nRecommendations:")
    print("=" * 50)
    for category, recommendation in recommendations.items():
        print(f"{category}: {recommendation}")
    
    print(f"\nChunked documents saved to: data/processed/chunks/")
    print("Files created:")
    print("  - financial_chunks_*.json (financial document chunks)")
    print("  - transcript_chunks_*.json (earnings call chunks)")
    print("  - chunks_metadata_*.csv (chunk metadata)")
    print("  - chunking_stats_*.json (chunking statistics)")