import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import PyPDF2
import pdfplumber
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    company: str
    quarter: str
    year: int
    document_type: str
    file_path: str
    text_length: str
    processing_timestamp: str

class FinancialTextCleaner:
    """
    A comprehensive processor for financial filings and earning call transcripts.
    """
    def __init__(self, data_dir: Path = Path("data")):
        self.base_dir = data_dir
        self.raw_dir = data_dir /"raw"
        self.processed_dir = data_dir / "processed"

        # Ensure directories exist
        self.processed_dir.mkdir(parents=True,exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # File parsing patterns
        self.file_pattern = re.compile(r'([A-Za-z0-9\s]+)\s+(q[1-4])\s+(\d{4})\.pdf', re.IGNORECASE)

        # Financial metrics extraction patterns
        self.financial_patterns = {
            'revenue_from_operations': [
                r'revenue\s+from\s+operations[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'total\s+revenue(?:\s+from\s+operations)?[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'net\s+sales[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'total_income': [
                r'total\s+income[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'total\s+revenue[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'net_profit': [
                r'net\s+profit(?:\s+for\s+the\s+period)?[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'profit\s+after\s+tax[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'basic_eps': [
                r'basic\s+earnings\s+per\s+share[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'basic.*?eps[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'earnings\s+per\s+share.*?re\.?\s*1[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'total_assets': [
                r'total\s+assets[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'assets\s+total[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'total_liabilities': [
                r'total\s+liabilities[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'liabilities\s+total[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'interim_equity_dividend': [
                r'interim\s+equity\s+dividend[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'interim.*?dividend[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'gross_margin': [
                r'gross\s+margin[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'gross\s+profit\s+margin[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'operating_margin': [
                r'operating\s+margin[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'operating\s+profit\s+margin[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'foods_business_revenue': [
                r'foods\s+business\s+revenue[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'food.*?revenue[^\d]*?(\d+(?:[\d,\.]*)?)'
            ],
            'premium_personal_care_contribution': [
                r'premium\s+personal\s+care\s+contribution[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'premium.*?personal.*?care.*?domestic.*?revenue[^\d]*?(\d+(?:[\d,\.]*)?)',
                r'personal\s+care.*?contribution[^\d]*?(\d+(?:[\d,\.]*)?)'
            ]
        }

    def extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from pdf")
            return ""

    
    def parse_filename(self, filename: str)->Optional[Tuple[str,str,int]]:
        """Parse filename to extract company, quarter, and year"""
        match = self.file_pattern.match(filename)
        if match:
            company = match.group(1).upper()
            quarter = match.group(2).upper()
            year = int(match.group(3))
            return company, quarter, year
        return None
    
    def clean_financial_filing_text(self, text: str) -> str:
        """Clean financial filing text for RAG processing."""
        # Remove page numbers and headers
        text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Normalize currency symbols
        text = re.sub(r'₹\s*', 'INR ', text)
        text = re.sub(r'\$\s*', 'USD ', text)
        
        # Clean up numbers (remove commas but preserve decimals)
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove regulatory boilerplate
        text = re.sub(r'pursuant to regulation.*?regulations, 2015', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    def clean_transcript_text(self, text: str) -> str:
        """Clean earnings call transcript text for RAG processing."""
        # Remove timestamps
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
        
        # Standardize speaker format
        text = re.sub(r'^([A-Z][A-Za-z\s]+):', r'\n\1:', text, flags=re.MULTILINE)
        
        # Remove moderator instructions
        text = re.sub(r'ladies and gentlemen.*?welcome to.*?earnings call', 
                     'Welcome to the earnings call', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page footers/headers
        text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Optional[float]]:
        """Extract key financial metrics from text."""
        metrics = {}
        text_lower = text.lower()
        
        for metric_name, patterns in self.financial_patterns.items():
            metric_value = None
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE | re.DOTALL)
                if matches:
                    try:
                        # Clean and convert the first match
                        number_str = matches[0].replace(',', '')
                        metric_value = float(number_str)
                        break
                    except (ValueError, IndexError):
                        continue
            
            metrics[metric_name] = metric_value
        
        return metrics
    
    def extract_transcript_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from earnings call transcripts."""
        sections = {}
        text_lower = text.lower()
        
        # Management Discussion
        mgmt_patterns = [
            r'(management.*?discussion|prepared\s+remarks|opening\s+statement)(.*?)(?=question|q\s*&\s*a|moderator)',
            r'(good\s+morning.*?managing\s+director)(.*?)(?=question|q\s*&\s*a)'
        ]
        
        for pattern in mgmt_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match and len(match.group(2).strip()) > 100:
                sections['management_discussion'] = match.group(2).strip()
                break
        
        # Q&A Section
        qa_patterns = [
            r'(question.*?answer|q\s*&\s*a\s+session|questions\s+from\s+participants)(.*?)$',
            r'(moderator.*?first\s+question)(.*?)$'
        ]
        
        for pattern in qa_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match and len(match.group(2).strip()) > 100:
                sections['qa_session'] = match.group(2).strip()
                break
        
        # If no specific sections found, use the full text
        if not sections:
            sections['full_transcript'] = text
        
        return sections

    def process_document(self, file_path: Path, doc_type: str) -> Optional[Dict[str, Any]]:
        """Process a single document (financial filing or transcript)."""
        try:
            # Parse filename
            parsed = self.parse_filename(file_path.name)
            if not parsed:
                self.logger.warning(f"Could not parse filename: {file_path.name}")
                return None
            
            company, quarter, year = parsed
            
            # Extract raw text
            raw_text = self.extract_pdf_text(file_path)
            if not raw_text or len(raw_text) < 100:
                self.logger.warning(f"No text extracted from {file_path.name}")
                return None
            
            # Clean text based on document type
            if doc_type == 'financial':
                cleaned_text = self.clean_financial_filing_text(raw_text)
                sections = {'full_document': cleaned_text}
                metrics = self.extract_financial_metrics(cleaned_text)
            else:  # transcript
                cleaned_text = self.clean_transcript_text(raw_text)
                sections = self.extract_transcript_sections(cleaned_text)
                metrics = {}
            
            # Create metadata
            metadata = DocumentMetadata(
                company=company,
                quarter=quarter,
                year=year,
                document_type=doc_type,
                file_path=str(file_path),
                text_length=len(cleaned_text),
                processing_timestamp=datetime.now().isoformat()
            )
            
            return {
                'metadata': metadata.__dict__,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'sections': sections,
                'extracted_metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def process_all_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """Process all documents in the raw directory."""
        results = {
            'financial': [],
            'transcript': []
        }
        
        # Process financial filings
        financial_dir = self.raw_dir / "financial"
        if financial_dir.exists():
            for file_path in financial_dir.glob("*.pdf"):
                self.logger.info(f"Processing financial filing: {file_path.name}")
                result = self.process_document(file_path, 'financial')
                if result:
                    results['financial'].append(result)
        
        # Process transcripts
        transcript_dir = self.raw_dir / "transcripts"
        if transcript_dir.exists():
            for file_path in transcript_dir.glob("*.pdf"):
                self.logger.info(f"Processing transcript: {file_path.name}")
                result = self.process_document(file_path, 'transcript')
                if result:
                    results['transcript'].append(result)
        
        return results

    def save_processed_documents(self, results: Dict[str, List[Dict[str, Any]]]):
        """Save processed documents to the processed directory."""
        
        for doc_type, documents in results.items():
            if not documents:
                continue
                
            # Create subdirectory
            output_dir = self.processed_dir / doc_type
            output_dir.mkdir(exist_ok=True)
            
            # Save individual documents
            metadata_list = []
            
            for doc in documents:
                metadata = doc['metadata']
                filename = f"{metadata['company']}_{metadata['quarter']}_{metadata['year']}.json"
                
                # Save full document
                with open(output_dir / filename, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                
                metadata_list.append(metadata)
            
            # Save metadata summary
            metadata_df = pd.DataFrame(metadata_list)
            metadata_df.to_csv(self.processed_dir / f"{doc_type}_metadata.csv", index=False)
            
            self.logger.info(f"Saved {len(documents)} {doc_type} documents to {output_dir}")
    def create_combined_dataset(self) -> pd.DataFrame:
        """Create a combined dataset from processed financial and transcript data."""
        combined_data = []
        
        # Load financial metadata
        financial_meta_path = self.processed_dir / "financial_metadata.csv"
        transcript_meta_path = self.processed_dir / "transcript_metadata.csv"
        
        if financial_meta_path.exists():
            financial_df = pd.read_csv(financial_meta_path)
        else:
            financial_df = pd.DataFrame()
            
        if transcript_meta_path.exists():
            transcript_df = pd.read_csv(transcript_meta_path)
        else:
            transcript_df = pd.DataFrame()
        
        # Merge datasets
        if not financial_df.empty and not transcript_df.empty:
            combined = pd.merge(
                financial_df, transcript_df,
                on=['company', 'quarter', 'year'],
                how='outer',
                suffixes=('_financial', '_transcript')
            )
        elif not financial_df.empty:
            combined = financial_df.copy()
        elif not transcript_df.empty:
            combined = transcript_df.copy()
        else:
            combined = pd.DataFrame()
        
        # Save combined dataset
        if not combined.empty:
            combined.to_csv(self.processed_dir / "combined_metadata.csv", index=False)
            self.logger.info(f"Created combined dataset with {len(combined)} records")
        
        return combined
    
    def generate_processing_report(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a comprehensive processing report."""
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'summary': {
                'financial_documents': len(results['financial']),
                'transcript_documents': len(results['transcript']),
                'total_documents': len(results['financial']) + len(results['transcript'])
            },
            'companies_processed': set(),
            'year_range': {'min': float('inf'), 'max': 0},
            'data_quality': {
                'financial': {},
                'transcript': {}
            }
        }
        
        # Analyze processed documents
        for doc_type, documents in results.items():
            for doc in documents:
                metadata = doc['metadata']
                report['companies_processed'].add(metadata['company'])
                report['year_range']['min'] = min(report['year_range']['min'], metadata['year'])
                report['year_range']['max'] = max(report['year_range']['max'], metadata['year'])
            
            # Calculate data quality metrics
            if documents:
                avg_text_length = np.mean([doc['metadata']['text_length'] for doc in documents])
                report['data_quality'][doc_type] = {
                    'average_text_length': avg_text_length,
                    'documents_with_sections': sum(1 for doc in documents if len(doc['sections']) > 1)
                }
        
        # Convert sets to lists for JSON serialization
        report['companies_processed'] = sorted(list(report['companies_processed']))
        
        # Handle edge case where no documents were processed
        if report['year_range']['min'] == float('inf'):
            report['year_range'] = {'min': None, 'max': None}
        
        # Save report
        with open(self.processed_dir / "processing_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Run the complete preprocessing pipeline."""
        self.logger.info("Starting financial text preprocessing pipeline...")
        
        # Process all documents
        results = self.process_all_documents()
        
        if not results['financial'] and not results['transcript']:
            self.logger.warning("No documents were successfully processed!")
            return {}
        
        # Save processed documents
        self.save_processed_documents(results)
        
        # Create combined dataset
        combined_df = self.create_combined_dataset()
        
        # Generate report
        report = self.generate_processing_report(results)
        
        self.logger.info("Preprocessing pipeline completed successfully!")
        self.logger.info(f"Processed {report['summary']['total_documents']} documents")
        self.logger.info(f"Companies: {', '.join(report['companies_processed'])}")
        
        return {
            'processed_results': results,
            'combined_dataframe': combined_df,
            'processing_report': report
        }

# Usage example and main execution
if __name__ == "__main__":
    cleaner = FinancialTextCleaner(data_dir=Path("data"))
    pipeline_results = cleaner.run_preprocessing_pipeline()
    
    print("\n" + "="*60)
    print("FINANCIAL TEXT PREPROCESSING COMPLETED")
    print("="*60)
    
    if pipeline_results:
        report = pipeline_results['processing_report']
        print(f"Total documents processed: {report['summary']['total_documents']}")
        print(f"Financial filings: {report['summary']['financial_documents']}")
        print(f"Transcripts: {report['summary']['transcript_documents']}")
        print(f"Companies: {', '.join(report['companies_processed'])}")
        print(f"Year range: {report['year_range']['min']} - {report['year_range']['max']}")
        print(f"\nProcessed files saved to: data/processed/")
    else:
        print("No documents were processed. Check your data/raw/ directory.")