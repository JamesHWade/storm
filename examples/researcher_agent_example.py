"""
Example demonstrating the Researcher agent in the Co-STORM framework.

This example shows how to:
1. Initialize a Co-STORM runner with a Researcher agent
2. Generate research ideas using the enhanced multi-stage approach:
   - Generate multiple brief ideas (returned as a proper list type)
   - Select and develop the most promising idea using rich context
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
    
    # Interactive notebook-style session
    print("\n"+"="*80)
    print("EXAMPLE 1: ENHANCED MULTI-STAGE RESEARCH IDEA GENERATION")
    print("="*80)
    print("Generating multiple research idea candidates (as a list) and selecting the best one...")
    print("This process now leverages much more conversation context for improved relevance.")
    
    # Generate research ideas using the new multi-stage approach
    research_idea = runner.generate_research_idea()
    
    print("\nRESEARCH IDEAS OUTPUT:")
    print("-"*50)
    print(research_idea)
    print("-"*50)
    print("\nThe output above contains:")
    print("1. A list of brief research idea candidates (5-10 one-sentence ideas)")
    print("2. The selected idea number and rationale for selection")
    print("3. The fully developed research idea")
    print("\nBehind the scenes, the IdeaCreation module now returns a proper list type")
    print("rather than a string, allowing for better handling of the candidates.")
    
    # The rest of the example
    print("\n"+"="*80)
    print("EXAMPLE 2: ASSESS THE RESEARCH IDEA")
    print("="*80)
    
    # Extract the developed idea for assessment
    idea_parts = research_idea.split("## Developed Research Idea:")
    if len(idea_parts) > 1:
        developed_idea = idea_parts[1].strip()
        print("\nAssessing the developed research idea...")
        assessment = runner.assess_research_idea(developed_idea)
    else:
        print("\nAssessing the complete research idea...")
        assessment = runner.assess_research_idea(research_idea)
    
    print("\nASSESSMENT:")
    print("-"*50)
    print(assessment)
    print("-"*50)
    
    print("\n"+"="*80)
    print("EXAMPLE 3: CREATE AN EXPERIMENTAL PLAN")
    print("="*80)
    
    print("\nCreating an experimental plan for the research idea...")
    plan = runner.create_experimental_plan(developed_idea, assessment)
    
    print("\nEXPERIMENTAL PLAN:")
    print("-"*50)
    print(plan)
    print("-"*50)
    
    print("\n"+"="*80)
    print("EXAMPLE 4: REFINE AN IDEA BASED ON FEEDBACK")
    print("="*80)
    
    print("\nRefining the research idea based on critical feedback...")
    feedback = """
    The idea has potential but needs to consider economic constraints for small-scale farmers.
    Additionally, more consideration should be given to implementation in regions with limited
    technological infrastructure. Consider adding elements of traditional farming knowledge.
    """
    
    refined_idea = runner.refine_research_idea(developed_idea, feedback)
    
    print("\nREFINED IDEA:")
    print("-"*50)
    print(refined_idea)
    print("-"*50)
    
    print("\n"+"="*80)
    print("EXAMPLE 5: COMPLETE RESEARCH PIPELINE WITH RICH CONTEXT")
    print("="*80)
    
    print("\nRunning the complete research pipeline with default settings (no refinement)...")
    print("This process now uses comprehensive conversation context rather than just the last utterance.")
    result = runner.research_pipeline(context="Investigate water conservation methods for drought conditions", add_to_conversation=True)
    
    print("\nRESEARCH PIPELINE RESULT COMPONENTS:")
    print("-"*50)
    print("1. Idea candidates (properly stored as a list internally)")
    print("2. Selected and developed idea")
    print("3. Assessment")
    print("4. Experimental plan")
    print("-"*50)
    
    print("\n"+"="*80)
    print("EXAMPLE 6: RESEARCH PIPELINE WITH AUTOMATIC IDEA REFINEMENT")
    print("="*80)
    
    print("\nRunning research pipeline with automatic idea refinement based on assessment...")
    result_with_refinement = runner.research_pipeline(
        context="Explore soil microbiome enhancement techniques",
        add_to_conversation=True,
        refine_idea_from_assessment=True
    )
    
    print("\nRESEARCH PIPELINE RESULT WITH REFINEMENT COMPONENTS:")
    print("-"*50)
    print("1. Original idea candidates and selection")
    print("2. Assessment")
    print("3. Refined idea (improved based on assessment)")
    print("4. Final experimental plan (based on refined idea)")
    print("-"*50)
    
    print("\n"+"="*80)
    print("CONVERSATION HISTORY")
    print("="*80)
    print(f"\nThe research results have been added to the conversation history.")
    print(f"Conversation now has {len(runner.conversation_history)} turns.")
    print("The system leverages this rich conversation history when generating new ideas.")

if __name__ == "__main__":
    main() 