"""
Minimal example to test the research_pipeline functionality
with the fixed import.
"""

import os
from knowledge_storm.collaborative_storm.engine import CoStormRunner, RunnerArgument, CollaborativeStormLMConfigs
from knowledge_storm.encoder import Encoder

def main():
    # Configure your API keys
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Create basic arguments
    args = RunnerArgument(
        topic="Veterinary medicine advances",
        retrieve_top_k=3,
        max_search_queries=2,
        node_expansion_trigger_count=5
    )
    
    # Initialize LM configuration
    lm_config = CollaborativeStormLMConfigs()
    lm_config.init(lm_type="openai")  # Use "azure" if using Azure OpenAI
    
    # Create encoder
    encoder = Encoder()
    
    # Create the runner
    runner = CoStormRunner(
        args=args,
        lm_config=lm_config,
        encoder=encoder
    )
    
    # Add a researcher
    runner.add_researcher(
        role_name="Veterinary Researcher",
        role_description="Expert in veterinary medicine with focus on geriatric care"
    )
    
    # Initialize with a simple warm start
    print("Performing warm start...")
    runner.warm_start()
    print("Warm start completed")
    
    # Test the research pipeline without refinement
    print("\nTesting research pipeline without refinement...")
    result1 = runner.research_pipeline(
        context="Non-invasive treatments for arthritis in elderly dogs",
        add_to_conversation=True
    )
    print("Research pipeline without refinement completed")
    
    # Test the research pipeline with refinement
    print("\nTesting research pipeline with refinement...")
    result2 = runner.research_pipeline(
        context="Nutritional approaches to cognitive decline in senior cats",
        add_to_conversation=True,
        refine_idea_from_assessment=True
    )
    print("Research pipeline with refinement completed")
    
    print("\nSuccess! The research pipeline is working correctly.")

if __name__ == "__main__":
    main() 