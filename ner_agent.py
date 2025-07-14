import spacy
import re
from typing import List, Dict

class NERAgent:
    """
    Agent that extracts named entities from renewable energy documents using spaCy.
    Focuses on dates, company names, monetary amounts, and contract durations.
    """
    def __init__(self):
        # Load the small English spaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def run(self, document: str) -> List[Dict[str, str]]:
        """
        Extract named entities from the document.
        Returns a list of entities with type and value.
        """
        doc = self.nlp(document)
        entities = []
        # Extract standard entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "DATE", "MONEY"]:
                entities.append({"type": ent.label_, "value": ent.text})
        # Heuristic for contract duration (e.g., '20 years', '5-year term')
        duration_matches = re.findall(r"\b(\d+\s*(?:years?|months?|days?))\b", document, re.IGNORECASE)
        for match in duration_matches:
            entities.append({"type": "DURATION", "value": match})
        return entities 