#!/usr/bin/env python3
"""
Example script showing how to use the BingSemanticScholarRetriever.

This script provides a simple demonstration of using the combined retriever
without relying on the full CoStormRunner framework.

You should set the following environment variables:
- BING_SEARCH_API_KEY: Your Bing Search API key
- SEMANTIC_SCHOLAR_API_KEY: (Optional) Your Semantic Scholar API key
- OPENAI_API_KEY: Your OpenAI API key for LLM functionality

Run with:
python examples/bing_semantic_scholar_costorm_example.py
"""

import os
import argparse

import dspy

# Import only what we need directly
from knowledge_storm.rm_combined import BingSemanticScholarRetriever


def load_api_key(toml_file_path=None):
    """Load API keys from environment variables or from a TOML file."""
    if toml_file_path and os.path.exists(toml_file_path):
        import toml
        with open(toml_file_path, "r") as f:
            secrets = toml.load(f)
            for k, v in secrets.items():
                if k not in os.environ:
                    os.environ[k] = v


def main():
    parser = argparse.ArgumentParser(description="Run a Bing and Semantic Scholar combined retriever example")
    parser.add_argument("--topic", type=str, default="Climate change mitigation strategies", 
                      help="Topic to research")
    parser.add_argument("--bing-weight", type=float, default=0.6, 
                      help="Weight for Bing results (0.0-1.0)")
    parser.add_argument("--k", type=int, default=10, 
                      help="Total number of results to retrieve")
    parser.add_argument("--start-year", type=int, default=None, 
                      help="Start year for Semantic Scholar search")
    parser.add_argument("--end-year", type=int, default=None, 
                      help="End year for Semantic Scholar search")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                      help="OpenAI model to use")
    args = parser.parse_args()
    
    # Load API keys
    load_api_key(toml_file_path="secrets.toml")
    
    # Check for required API keys
    if not os.getenv("BING_SEARCH_API_KEY"):
        print("Error: BING_SEARCH_API_KEY environment variable is required")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        return
    
    # Configure dspy with OpenAI
    dspy.settings.configure(lm=dspy.OpenAI(model=args.model, api_key=os.getenv("OPENAI_API_KEY")))
    
    # Create the combined retriever
    retriever = BingSemanticScholarRetriever(
        k=args.k,
        bing_weight=args.bing_weight,
        semantic_scholar_kwargs={
            "start_year": args.start_year,
            "end_year": args.end_year,
        }
    )
    
    # Query the retriever
    print(f"\nResearching topic: {args.topic}")
    print(f"Weight: {args.bing_weight*100:.0f}% Bing, {(1-args.bing_weight)*100:.0f}% Semantic Scholar")
    print(f"Retrieving {args.k} total results...\n")
    
    results = retriever(args.topic)
    
    # Display results
    print(f"Retrieved {len(results)} results:")
    for i, result in enumerate(results, 1):
        # Determine source
        source = "Unknown"
        if hasattr(result, "url") and "semanticscholar" in getattr(result, "url", ""):
            source = "Semantic Scholar"
        elif hasattr(result, "url"):
            source = "Bing"
        
        # Print result details
        print(f"\n--- Result {i} ({source}) ---")
        if hasattr(result, "title"):
            print(f"Title: {result.title}")
        if hasattr(result, "authors") and result.authors:
            print(f"Authors: {result.authors}")
        if hasattr(result, "year") and result.year:
            print(f"Year: {result.year}")
        if hasattr(result, "url"):
            print(f"URL: {result.url}")
        if hasattr(result, "text"):
            # Truncate long text for display
            text = result.text[:300] + "..." if len(result.text) > 300 else result.text
            print(f"Content: {text}")
    
    # Use dspy's LM to generate a summary
    print("\n--- Generating summary with LLM ---")
    
    # Format the content for the LLM
    context = "\n\n".join([f"Source {i+1}: {r.text}" for i, r in enumerate(results)])
    prompt = f"""
    Please create a comprehensive summary on the topic: {args.topic}
    
    Use the following information retrieved from Bing and Semantic Scholar:
    
    {context}
    
    Generate a detailed summary that integrates the key points from these sources.
    """
    
    # Generate the summary using dspy
    lm = dspy.get_lm()
    response = lm(prompt)
    
    print(f"\n{response}")
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
