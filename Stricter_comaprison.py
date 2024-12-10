"""
Strict Drug Comparison Analyzer

This script performs a detailed comparison between DrugBank and MARMAR-generated drug information using ClinicalBERT 
embeddings and a set of strict metrics. It provides insights into the semantic, technical, and content-based alignment 
between these two sources for each drug.

Main Features:
1. Integrates ClinicalBERT for sentence embeddings and semantic analysis.
2. Implements strict comparison metrics:
   - Semantic Similarity: Embedding-based similarity.
   - Technical Term Overlap: Overlap of predefined technical terms.
   - Content Coverage: Completeness and length comparison.
3. Analyzes fields such as "mechanism of action," "pharmacodynamics," and "safety."
4. Generates detailed results for each drug, including a weighted score, and aggregates statistics across all drugs.
5. Saves results to a JSON file and logs detailed statistics for review.

Inputs:
- drug_validation_results.json: Contains validated DrugBank information for drugs.
- marmar_analyses.json: Contains MARMAR-generated data for the same drugs.

Outputs:
- strict_comparison_results.json: Contains detailed and summary results of the comparison.
"""


import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple
import json
import logging
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import re

class StrictDrugComparisonAnalyzer:
    def __init__(self):
        """Initialize analyzer with ClinicalBERT"""
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define technical terms and important keywords
        self.technical_terms = set([
            'receptor', 'enzyme', 'inhibitor', 'agonist', 'antagonist',
            'metabolism', 'clearance', 'bioavailability', 'half-life',
            'cytochrome', 'transporter', 'binding', 'absorption',
            'pharmacokinetic', 'pharmacodynamic', 'excretion', 'plasma',
            'concentration', 'therapeutic', 'toxicity', 'adverse', 'interaction'
        ])
        
        # Load data
        self._load_data()

    def _load_data(self):
        """Load and validate input data"""
        try:
            with open('drug_validation_results.json', 'r') as f:
                self.drugbank_data = json.load(f)
            with open('marmar_analyses.json', 'r') as f:
                self.marmar_data = json.load(f)
            logging.info("Successfully loaded datasets")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Split on periods followed by spaces and capital letters
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_sentence_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for each sentence in text"""
        sentences = self.split_sentences(text)
        embeddings = []
        
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(sentence, return_tensors="pt", 
                                      padding=True, truncation=True, 
                                      max_length=256).to(self.device)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :])
        
        return torch.cat(embeddings, dim=0) if embeddings else None

    def extract_technical_terms(self, text: str) -> set:
        """Extract technical terms from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words).intersection(self.technical_terms)

    def calculate_strict_similarity(self, text1: str, text2: str) -> Dict:
        """Calculate similarity using multiple strict metrics"""
        if not text1 or not text2:
            return {
                'semantic_similarity': 0.0,
                'technical_term_overlap': 0.0,
                'content_coverage': 0.0,
                'weighted_score': 0.0
            }

        # Get embeddings
        emb1 = self.get_sentence_embeddings(text1)
        emb2 = self.get_sentence_embeddings(text2)
        
        if emb1 is None or emb2 is None:
            return {
                'semantic_similarity': 0.0,
                'technical_term_overlap': 0.0,
                'content_coverage': 0.0,
                'weighted_score': 0.0
            }

        # 1. Semantic Similarity
        emb1_mean = torch.mean(emb1, dim=0).cpu().numpy().reshape(1, -1)
        emb2_mean = torch.mean(emb2, dim=0).cpu().numpy().reshape(1, -1)
        semantic_sim = float(cosine_similarity(emb1_mean, emb2_mean)[0, 0])

        # 2. Technical Term Overlap
        terms1 = self.extract_technical_terms(text1)
        terms2 = self.extract_technical_terms(text2)
        
        if terms1 or terms2:
            term_overlap = len(terms1.intersection(terms2)) / len(terms1.union(terms2)) if terms1.union(terms2) else 0.0
        else:
            term_overlap = 0.0

        # 3. Content Coverage
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Compare length and information density
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        word_overlap = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0.0
        content_coverage = (len_ratio + word_overlap) / 2

        # 4. Calculate Weighted Score with stricter weights
        weighted_score = (
            semantic_sim * 0.35 +          # Semantic understanding
            term_overlap * 0.40 +          # Technical accuracy (increased weight)
            content_coverage * 0.25        # Completeness
        )

        return {
            'semantic_similarity': round(semantic_sim, 3),
            'technical_term_overlap': round(term_overlap, 3),
            'content_coverage': round(content_coverage, 3),
            'weighted_score': round(weighted_score, 3)
        }

    def compare_drug_fields(self, drug_name: str) -> Dict:
        """Compare DrugBank and MARMAR fields with strict metrics"""
        drugbank_entry = next((d for d in self.drugbank_data['found_drugs'] 
                             if d['name'] == drug_name), None)
        marmar_entry = self.marmar_data.get(drug_name)
        
        if not drugbank_entry or not marmar_entry:
            logging.warning(f"Missing data for drug: {drug_name}")
            return None

        comparisons = {
            'mechanism': {
                'drugbank_text': drugbank_entry['drugbank_info']['mechanism_of_action'],
                'marmar_text': marmar_entry['pharmacologicalExplanation'],
            },
            'effects': {
                'drugbank_text': drugbank_entry['drugbank_info']['pharmacodynamics'],
                'marmar_text': marmar_entry['detailedExplanation'],
            },
            'safety': {
                'drugbank_text': f"{drugbank_entry['drugbank_info']['toxicity']} {drugbank_entry['drugbank_info'].get('drug_interactions', '')}",
                'marmar_text': f"{marmar_entry['generalAdvice']} {marmar_entry.get('interactionRiskLevel', '')}",
            }
        }
        
        results = {}
        for field, texts in comparisons.items():
            results[field] = self.calculate_strict_similarity(
                texts['drugbank_text'], 
                texts['marmar_text']
            )
            
        return {
            'drug_name': drug_name,
            'detailed_scores': results,
            'average_weighted_score': round(np.mean([
                r['weighted_score'] for r in results.values()
            ]), 3)
        }

    def analyze_all_drugs(self) -> Dict:
        """Analyze all drugs with strict comparison metrics"""
        results = {}
        aggregated_scores = {
            'mechanism': {
                'semantic': [], 'technical': [], 'coverage': [], 'weighted': []
            },
            'effects': {
                'semantic': [], 'technical': [], 'coverage': [], 'weighted': []
            },
            'safety': {
                'semantic': [], 'technical': [], 'coverage': [], 'weighted': []
            }
        }
        
        for drug_name in tqdm(self.marmar_data.keys(), desc="Analyzing drugs"):
            result = self.compare_drug_fields(drug_name)
            if result:
                results[drug_name] = result
                
                # Aggregate scores for statistics
                for field, scores in result['detailed_scores'].items():
                    aggregated_scores[field]['semantic'].append(scores['semantic_similarity'])
                    aggregated_scores[field]['technical'].append(scores['technical_term_overlap'])
                    aggregated_scores[field]['coverage'].append(scores['content_coverage'])
                    aggregated_scores[field]['weighted'].append(scores['weighted_score'])

        # Calculate summary statistics
        summary = {field: {
            metric: {
                'mean': round(np.mean(scores), 3),
                'std': round(np.std(scores), 3),
                'min': round(min(scores), 3),
                'max': round(max(scores), 3)
            }
            for metric, scores in field_scores.items()
        } for field, field_scores in aggregated_scores.items()}

        return {
            'detailed_results': results,
            'summary_statistics': summary
        }

    def save_results(self, results: Dict, filename: str = 'strict_comparison_results.json'):
        """Save detailed analysis results"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")

def main():
    analyzer = StrictDrugComparisonAnalyzer()
    results = analyzer.analyze_all_drugs()
    analyzer.save_results(results)
    
    # Print detailed summary
    print("\nStrict Analysis Summary:")
    for field, metrics in results['summary_statistics'].items():
        print(f"\n{field.upper()} Comparison:")
        for metric, stats in metrics.items():
            print(f"{metric}:")
            print(f"  Mean: {stats['mean']:.3f} (±{stats['std']:.3f})")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")

if __name__ == "__main__":
    main()