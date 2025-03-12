#!/usr/bin/env python3
"""
Simple test script for the SemanticScholarRM retriever.
This script performs a basic search using the Semantic Scholar API and displays the results.

You can run this script directly without API keys, but you'll get better results and higher
rate limits if you set the SEMANTIC_SCHOLAR_API_KEY environment variable.
"""

import os
import json
from knowledge_storm.rm import SemanticScholarRM
from knowledge_storm.utils import load_api_key

def main():
    # Load API keys if available (optional for Semantic Scholar)
    try:
        load_api_key(toml_file_path="secrets.toml")
        print("Loaded API keys from secrets.toml")
    except Exception as e:
        print(f"Note: Could not load API keys: {e}")
        print("Proceeding without API key (limited rate limits)")
    
    # Initialize the retriever
    retriever = SemanticScholarRM(
        k=5,  # Number of results to retrieve
        # Optional filters for publication years
        # year_filter={"start_year": 2020, "end_year": 2023}
    )
    
    # Test query
    test_query = input("Enter a research topic to search (e.g., 'large language models'): ")
    if not test_query:
        test_query = "large language models"
        print(f"Using default query: '{test_query}'")
    
    print(f"\nSearching for: '{test_query}'")
    print("This may take a few seconds...\n")
    
    # Perform the search
    results = retriever.forward(test_query)
    
    # Display results
    if not results:
        print("No results found.")
    else:
        print(f"Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            
            # Print first part of abstract if available
            if result['snippets'] and len(result['snippets']) > 0:
                abstract = result['snippets'][0]
                print(f"Abstract: {abstract[:150]}..." if len(abstract) > 150 else f"Abstract: {abstract}")
            
            # Print publication info if available
            if len(result['snippets']) > 1:
                pub_info = result['snippets'][1]
                print(f"Publication info: {pub_info}")
            
            print("-" * 80)
    
    # Print usage statistics
    usage_stats = retriever.get_usage_and_reset()
    print(f"Usage statistics: {json.dumps(usage_stats, indent=2)}")

if __name__ == "__main__":
    main()
