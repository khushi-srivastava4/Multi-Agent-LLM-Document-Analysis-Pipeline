from ner_agent import NERAgent
from classifier_agent import DocumentClassifierAgent
from anomaly_agent import AnomalyDetectorAgent
from typing import Dict, Any, List
import logging

class MultiAgentPipeline:
    """
    Orchestrates the NER, classification, and anomaly detection agents.
    Provides logging and a summary report function.
    """
    def __init__(self, openai_api_key: str):
        self.ner_agent = NERAgent()
        self.classifier_agent = DocumentClassifierAgent(openai_api_key)
        self.anomaly_agent = AnomalyDetectorAgent(openai_api_key)
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MultiAgentPipeline")

    def run(self, document: str) -> Dict[str, Any]:
        """
        Run the document through the NER, classification, and anomaly detection agents.
        Returns a dictionary with entities, document type, and anomalies.
        """
        self.logger.info("Starting NER agent...")
        entities = self.ner_agent.run(document)
        self.logger.info(f"Entities extracted: {entities}")

        self.logger.info("Starting Document Classifier agent...")
        doc_type = self.classifier_agent.run(document)
        self.logger.info(f"Document classified as: {doc_type}")

        self.logger.info("Starting Anomaly Detector agent...")
        anomalies = self.anomaly_agent.run(document, entities, doc_type)
        self.logger.info(f"Anomalies detected: {anomalies}")

        return {
            "entities": entities,
            "doc_type": doc_type,
            "anomalies": anomalies
        }

    def summary_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a summary report from the pipeline result.
        """
        report = [
            "==== Document Analysis Summary ===="
        ]
        report.append(f"Document Type: {result['doc_type']}")
        report.append("\nNamed Entities:")
        for ent in result["entities"]:
            report.append(f"- {ent['type']}: {ent['value']}")
        report.append("\nAnomalies or Missing Info:")
        if result["anomalies"]:
            for a in result["anomalies"]:
                report.append(f"- {a}")
        else:
            report.append("None")
        return "\n".join(report) 