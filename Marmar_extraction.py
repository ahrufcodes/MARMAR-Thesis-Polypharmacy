#This code basically loads a JSON file that contains validated drug data.
#It then uses the OpenAI API to generate a technical analysis for each drug.
#The results are saved to a new JSON file, called marmar_analyses.json.

import json
import time
import logging
import os
from openai import OpenAI
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential


class MarmarTechnicalValidator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize validator with OpenAI client and setup logging
        
        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, will try to get from environment
        """
        # Initialize OpenAI client with explicit API key handling
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided either through constructor or OPENAI_API_KEY environment variable")
            self.client = OpenAI(api_key=api_key)
        
        # Setup logging with both file and console handlers
        self.setup_logging()
        
        # Load validated drugs
        self.validated_drugs = self.load_validated_drugs()

    def setup_logging(self):
        """Setup logging configuration with both file and console handlers"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('marmar_validation.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
        logger.addHandler(console_handler)

    def load_validated_drugs(self) -> Dict:
        """Load and validate the drug validation results"""
        try:
            with open('drug_validation_results.json', 'r') as f:
                validated_drugs = json.load(f)
            
            if not isinstance(validated_drugs, dict) or 'found_drugs' not in validated_drugs:
                raise ValueError("Invalid drug validation results format")
                
            drug_count = len(validated_drugs['found_drugs'])
            logging.info(f"Successfully loaded {drug_count} validated drugs")
            return validated_drugs
            
        except FileNotFoundError:
            logging.error("drug_validation_results.json not found")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing drug validation results: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading drugs: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_analysis(self, drug_data: Dict) -> Optional[Dict]:
        """
        Generate MARMAR analysis for a single drug with improved error handling
        
        Args:
            drug_data (Dict): Drug data containing name and technical information
            
        Returns:
            Optional[Dict]: Analysis results or None if generation fails
        """
        try:
            # Validate required drug data fields
            required_fields = ['name', 'drugbank_info']
            if not all(field in drug_data for field in required_fields):
                raise ValueError(f"Missing required fields in drug data: {required_fields}")

            technical_info = f"""Medications: {drug_data['name']}

Technical Information:
Mechanism of Action: {drug_data['drugbank_info'].get('mechanism_of_action', 'Not available')}
Pharmacodynamics: {drug_data['drugbank_info'].get('pharmacodynamics', 'Not available')}
Known Drug Interactions: {drug_data['drugbank_info'].get('drug_interactions', 'Not available')}
Toxicity Information: {drug_data['drugbank_info'].get('toxicity', 'Not available')}"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI-powered pharmacist assistant with expertise in drug interactions, pharmacology, and personalized healthcare. Your knowledge is based on peer-reviewed research and official drug databases.

Your responses must always be in valid JSON format with the following structure:
{
    "interactionRiskLevel": "SEVERE" | "MODERATE" | "MILD",
    "generalExplanation": "string",
    "detailedExplanation": "string",
    "generalAdvice": "string",
    "pharmacologicalExplanation": "string",
    "references": ["string"],
    "alternativeMedications": "string or null",
    "dietaryPrecautions": [
        {
            "name": "string",
            "explanation": "string"
        }
    ]
}"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the following medication and its technical information. Respond only with a JSON object following the specified structure:\n\n{technical_info}"
                    }
                ],
                temperature=0.3,
                max_tokens=2048
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result.update({
                '_metadata': {
                    'drug_name': drug_data['name'],
                    'drugbank_id': drug_data['drugbank_info'].get('drugbank_id', 'Unknown'),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_version': '2.0'
                }
            })
            
            return result
            
        except ValueError as e:
            logging.error(f"Validation error for {drug_data.get('name', 'Unknown drug')}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Analysis failed for {drug_data.get('name', 'Unknown drug')}: {str(e)}")
            return None

    def process_drugs(self) -> Dict[str, Dict]:
        """
        Process all validated drugs with progress tracking
        
        Returns:
            Dict[str, Dict]: Dictionary of drug analyses
        """
        analyses = {}
        total_drugs = len(self.validated_drugs['found_drugs'])
        
        for idx, drug in enumerate(self.validated_drugs['found_drugs'], 1):
            drug_name = drug.get('name', 'Unknown drug')
            print(f"\nProcessing {idx}/{total_drugs}: {drug_name}")
            
            # Add delay between API calls to avoid rate limiting
            if idx > 1:
                time.sleep(1)
            
            result = self.generate_analysis(drug)
            if result:
                analyses[drug_name] = result
                logging.info(f"Completed analysis for {drug_name}")
            else:
                logging.warning(f"Failed analysis for {drug_name}")

        return analyses

    def save_results(self, analyses: Dict[str, Dict], filename: str = 'marmar_analyses.json'):
        """
        Save analyses to JSON file with backup
        
        Args:
            analyses (Dict[str, Dict]): Analysis results to save
            filename (str): Output filename
        """
        if not analyses:
            logging.warning("No analyses to save")
            return

        try:
            # Create backup of existing file if it exists
            if os.path.exists(filename):
                backup_name = f"{filename}.backup.{int(time.time())}"
                os.rename(filename, backup_name)
                logging.info(f"Created backup: {backup_name}")

            with open(filename, 'w') as f:
                json.dump(analyses, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved {len(analyses)} analyses to {filename}")
            print(f"\nAnalyses saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving analyses: {str(e)}")
            print(f"Error saving results: {str(e)}")

def main():
    """Main execution function with comprehensive error handling"""
    try:
        # First try to get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        # If not in environment, prompt user
        if not api_key:
            api_key = input("Please enter your OpenAI API key: ").strip()
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            # Optionally save to environment for current session
            os.environ["OPENAI_API_KEY"] = api_key
        
        validator = MarmarTechnicalValidator(api_key)
        analyses = validator.process_drugs()
        validator.save_results(analyses)
        print(f"\nCompleted analyses for {len(analyses)} drugs")
        
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        logging.error(f"Configuration error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logging.error(f"Execution error: {str(e)}")

if __name__ == "__main__":
    main()