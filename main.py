from multi_agent_pipeline import MultiAgentPipeline
import getpass
import os

SAMPLE_FILE = "sample_document.txt"


def ensure_sample_file():
    """
    Ensure a sample .txt file exists. If not, create one with a sample renewable energy document.
    """
    if not os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
            f.write(
                "This Power Purchase Agreement (PPA) is made on January 1, 2023, between Green Energy Corp, located in California, and Solar Solutions Inc. The agreement covers the supply of 100MW solar power for 20 years. The contract value is $50 million."
            )


def read_document(path: str) -> str:
    """
    Read the document text from a .txt file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    """
    Main entry point: loads API key, reads document, runs pipeline, prints summary report.
    """
    print("Enter your OpenAI API key:")
    openai_api_key = getpass.getpass()

    ensure_sample_file()
    print(f"\nReading sample document from '{SAMPLE_FILE}'...")
    document = read_document(SAMPLE_FILE)

    pipeline = MultiAgentPipeline(openai_api_key)
    print("\nProcessing document through multi-agent pipeline...\n")
    result = pipeline.run(document)

    print(pipeline.summary_report(result))

if __name__ == "__main__":
    main() 