from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
import re

class AnomalyDetectorAgent:
    """
    Agent that checks for missing or unusual fields in renewable energy documents.
    Uses LLM for flexible anomaly detection and rule-based logic as fallback.
    """
    def __init__(self, openai_api_key: str):
        # Initialize the LLM and prompt for anomaly detection
        self.llm = OpenAI(openai_api_key=openai_api_key)
        self.prompt = PromptTemplate(
            input_variables=["document", "entities", "doc_type"],
            template=(
                "You are an expert in renewable energy contracts and documents.\n"
                "Given the document, its extracted entities, and its type ({doc_type}), list any anomalies or missing critical fields.\n"
                "Check for missing effective date, payment clause, unrealistic values, or other issues.\n"
                "If none, say 'None'.\n\nDocument:\n{document}\n\nEntities:\n{entities}\n\nAnomalies:"
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def rule_based_check(self, document: str, entities: List[Dict[str, str]], doc_type: str) -> List[str]:
        """
        Rule-based fallback for anomaly detection.
        Checks for missing effective date, payment clause, and unrealistic values.
        """
        anomalies = []
        # Check for effective date
        has_date = any(e['type'] == 'DATE' for e in entities)
        if not has_date:
            anomalies.append("Missing effective date.")
        # Check for payment clause (simple keyword search)
        if not re.search(r'payment|payable|amount due|invoice', document, re.IGNORECASE):
            anomalies.append("Missing payment clause or amount.")
        # Check for unrealistic values (e.g., $0, negative, or very high)
        for e in entities:
            if e['type'] == 'MONEY':
                if re.search(r'\$0|zero|negative', e['value'], re.IGNORECASE):
                    anomalies.append(f"Unrealistic value: {e['value']}")
                if re.search(r'\$\s*\d{9,}', e['value']):
                    anomalies.append(f"Suspiciously high value: {e['value']}")
        return anomalies

    def run(self, document: str, entities: List[Dict[str, str]], doc_type: str) -> List[str]:
        """
        Detect anomalies using LLM and rule-based fallback.
        Returns a list of anomaly descriptions.
        """
        # Prepare entities as a string for LLM
        entities_str = str(entities)
        # Run LLM-based anomaly detection
        llm_result = self.chain.run(document=document, entities=entities_str, doc_type=doc_type)
        # If LLM returns 'None' or empty, use rule-based fallback
        if not llm_result or llm_result.strip().lower() == 'none':
            return self.rule_based_check(document, entities, doc_type)
        # Otherwise, parse LLM output into a list
        return [a.strip() for a in llm_result.split('\n') if a.strip() and a.strip().lower() != 'none'] 