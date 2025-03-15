"""
Example demonstrating the Researcher agent in the Co-STORM framework.

This example shows how to:
1. Initialize a Co-STORM runner with a Researcher agent
2. Generate research ideas
3. Assess research ideas
4. Create experimental plans
5. Refine ideas based on feedback
6. Use the complete research pipeline and add results to conversation
"""

import os
from knowledge_storm.collaborative_storm.engine import CoStormRunner, RunnerArgument, CollaborativeStormLMConfigs
from knowledge_storm.encoder import Encoder

# Set your API keys in environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # For OpenAI
# For Azure: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
# For Bing search: BING_SEARCH_API_KEY

def main():
    print("Initializing Co-STORM runner with Researcher agent...")
    
    # Create runner arguments
    args = RunnerArgument(
        topic="Climate-resistant agriculture techniques",
        retrieve_top_k=5,
        max_search_queries=3,
        node_expansion_trigger_count=5  # Required parameter for KnowledgeBase
    )
    
    # Initialize language model configuration
    lm_config = CollaborativeStormLMConfigs()
    lm_config.init(lm_type="openai", temperature=0.7)  # Use "azure" if using Azure OpenAI
    
    # Create encoder
    encoder = Encoder()
    
    # Create the Co-STORM runner
    runner = CoStormRunner(
        args=args,
        lm_config=lm_config,
        encoder=encoder
    )
    
    # Add a specialized researcher to the discussion
    researcher_index = runner.add_researcher(
        role_name="Agricultural Innovation Scientist", 
        role_description="Specializes in climate-adaptive farming techniques and sustainable agriculture"
    )
    print(f"Added researcher at index {researcher_index}")
    
    # Optional: Initialize with background knowledge
    print("\nPerforming warm start to gather background knowledge...")
    runner.warm_start()
    print("Warm start completed")
    
    # Example 1: Generate a research idea
    print("\n=== EXAMPLE 1: GENERATING A RESEARCH IDEA ===\n")
    idea = runner.generate_research_idea(
        context="Consider how combining traditional agricultural knowledge with modern technology could improve resilience"
    )
    print(idea)
    
    # Example 2: Assess the idea
    print("\n=== EXAMPLE 2: ASSESSING THE RESEARCH IDEA ===\n")
    assessment = runner.assess_research_idea(idea)
    print(assessment)
    
    # Example 3: Gather feedback and refine the idea
    print("\n=== EXAMPLE 3: REFINING THE IDEA BASED ON FEEDBACK ===\n")
    feedback = """
    The idea has potential but needs more specific implementation details. 
    Consider how it could be applied in developing countries with limited resources.
    Also, address potential scaling challenges and how to measure success metrics.
    """
    refined_idea = runner.refine_research_idea(idea, feedback)
    print("ORIGINAL IDEA:")
    print(idea)
    print("\nFEEDBACK:")
    print(feedback)
    print("\nREFINED IDEA:")
    print(refined_idea)
    
    # Example 4: Create an experimental plan for the refined idea
    print("\n=== EXAMPLE 4: CREATING AN EXPERIMENTAL PLAN ===\n")
    # Generate a new assessment for the refined idea
    refined_assessment = runner.assess_research_idea(refined_idea)
    plan = runner.create_experimental_plan(refined_idea, refined_assessment)
    print(plan)
    
    # Example 5: Run complete research pipeline and add to conversation
    print("\n=== EXAMPLE 5: COMPLETE RESEARCH PIPELINE WITH CONVERSATION INTEGRATION ===\n")
    print("Before research pipeline, conversation history has", len(runner.conversation_history), "turns")
    
    research_output = runner.research_pipeline(
        context="Novel approaches to water conservation in drought-prone regions",
        add_to_conversation=True
    )
    
    print("After research pipeline, conversation history has", len(runner.conversation_history), "turns")
    
    # Example 6: Run research pipeline with automatic idea refinement
    print("\n=== EXAMPLE 6: RESEARCH PIPELINE WITH AUTOMATIC IDEA REFINEMENT ===\n")
    
    refined_research_output = runner.research_pipeline(
        context="Sustainable energy solutions for rural communities",
        add_to_conversation=True,
        refine_idea_from_assessment=True  # This will refine the idea based on the assessment
    )
    
    print("INITIAL IDEA:")
    print(refined_research_output["idea"])
    print("\nASSESSMENT:")
    print(refined_research_output["assessment"])
    print("\nREFINED IDEA:")
    print(refined_research_output["refined_idea"])
    print("\nEXPERIMENTAL PLAN:")
    print(refined_research_output["plan"])
    
    # Add a follow-up question to the conversation
    print("\n=== EXAMPLE 7: INTERACTIVE CONVERSATION WITH RESEARCHER ===\n")
    runner.step(user_utterance="What are the biggest challenges in implementing the water conservation techniques you proposed?")
    
    # Let the system respond (might be the researcher or another expert)
    response = runner.step()
    print(f"Response from {response.role}:")
    print(response.utterance)
    
    print("\nResearch investigation complete!")

if __name__ == "__main__":
    main() 