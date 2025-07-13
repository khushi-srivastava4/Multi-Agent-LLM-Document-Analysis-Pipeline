from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DocumentClassifierAgent:
    """
    Agent that classifies renewable energy-related documents as one of:
    - Power Purchase Agreement (PPA)
    - Invoice
    - Contract Summary
    Uses a pre-trained LLM (e.g., OpenAI GPT) via LangChain.
    """
    def __init__(self, openai_api_key: str):
        # Initialize the LLM and prompt for classification
        self.llm = OpenAI(openai_api_key=openai_api_key)
        self.prompt = PromptTemplate(
            input_variables=["document"],
            template=(
                "You are an expert in renewable energy legal and financial documents.\n"
                "Classify the following document as one of the following types: 'Power Purchase Agreement (PPA)', 'Invoice', or 'Contract Summary'.\n"
                "If unsure, choose the closest match.\n"
                "Document:\n{document}\n\nType (choose one: Power Purchase Agreement (PPA), Invoice, Contract Summary):"
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, document: str) -> str:
        """
        Classify the document using the LLM. Returns the document type as a string.
        """
        # Run the LLM chain with the document as input
        result = self.chain.run(document=document)
        return result.strip() 