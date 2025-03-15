"""
Interactive Notebook-Style Example: Research Pipeline with Idea Refinement

This example demonstrates how to use the Co-STORM Researcher agent in an interactive notebook-style workflow,
focusing on the research pipeline with automatic idea refinement based on assessment.
"""

import os
from knowledge_storm.collaborative_storm.engine import CoStormRunner, RunnerArgument, CollaborativeStormLMConfigs
from knowledge_storm.encoder import Encoder

# -------------------------------------------------------------------------
# Cell 1: Initialize the Co-STORM system
# -------------------------------------------------------------------------

# Set API keys in environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Setup runner
args = RunnerArgument(
    topic="Next-generation renewable energy storage",
    retrieve_top_k=5,
    max_search_queries=3,
    node_expansion_trigger_count=5
)

# Initialize LM config
lm_config = CollaborativeStormLMConfigs()
lm_config.init(lm_type="openai")  # Use "azure" if using Azure OpenAI

# Create encoder and runner
encoder = Encoder()
runner = CoStormRunner(
    args=args,
    lm_config=lm_config,
    encoder=encoder
)

# -------------------------------------------------------------------------
# Cell 2: Add a specialized researcher and warm start
# -------------------------------------------------------------------------

# Add an energy storage specialist researcher
researcher_index = runner.add_researcher(
    role_name="Energy Storage Specialist",
    role_description="Expert in cutting-edge energy storage technologies and grid integration strategies"
)
print(f"Added researcher at index {researcher_index}")

# Perform warm start to gather background knowledge
print("\nPerforming warm start to gather background knowledge...")
runner.warm_start()
print("Warm start completed")

# -------------------------------------------------------------------------
# Cell 3: Simple research pipeline (without refinement)
# -------------------------------------------------------------------------

print("\n=== STANDARD RESEARCH PIPELINE ===\n")

# Run the standard research pipeline
standard_output = runner.research_pipeline(
    context="Seasonal thermal energy storage for residential buildings",
    add_to_conversation=True,
    refine_idea_from_assessment=False  # Default behavior - no refinement
)

print("RESEARCH IDEA:")
print(standard_output["idea"][:150] + "...")  # Show just the beginning for brevity
print("\nASSESSMENT (excerpt):")
print(standard_output["assessment"][:150] + "...")
print("\nEXPERIMENTAL PLAN (excerpt):")
print(standard_output["plan"][:150] + "...")

# -------------------------------------------------------------------------
# Cell 4: Research pipeline with automatic idea refinement
# -------------------------------------------------------------------------

print("\n=== RESEARCH PIPELINE WITH IDEA REFINEMENT ===\n")

# Run the research pipeline with idea refinement
refined_output = runner.research_pipeline(
    context="Grid-scale gravitational energy storage alternatives to pumped hydro",
    add_to_conversation=True,
    refine_idea_from_assessment=True  # Enable automatic refinement
)

print("INITIAL IDEA:")
print(refined_output["idea"])

print("\nASSESSMENT:")
print(refined_output["assessment"])

print("\nREFINED IDEA (based on assessment feedback):")
print(refined_output["refined_idea"])

print("\nEXPERIMENTAL PLAN (for the refined idea):")
print(refined_output["plan"])

# -------------------------------------------------------------------------
# Cell 5: Manual refinement process with explicit feedback
# -------------------------------------------------------------------------

print("\n=== MANUAL REFINEMENT PROCESS ===\n")

# Generate a new research idea
idea = runner.generate_research_idea(
    context="Solid-state batteries for grid-scale energy storage"
)
print("ORIGINAL IDEA:")
print(idea)

# Assess the idea
assessment = runner.assess_research_idea(idea)
print("\nASSESSMENT:")
print(assessment)

# Extract specific feedback points from the assessment
specific_feedback = """
Based on the assessment, the idea needs improvement in these areas:
1. Consider manufacturing scalability challenges
2. Address cost concerns for grid-scale deployment
3. Provide more detail on safety improvements compared to existing technologies
4. Consider timeline realistic for commercial implementation
"""
print("\nEXTRACTED FEEDBACK:")
print(specific_feedback)

# Refine the idea with specific feedback
refined_idea = runner.refine_research_idea(idea, specific_feedback)
print("\nREFINED IDEA:")
print(refined_idea)

# Create an experimental plan for the refined idea
plan = runner.create_experimental_plan(refined_idea, assessment)
print("\nEXPERIMENTAL PLAN:")
print(plan)

# -------------------------------------------------------------------------
# Cell 6: Interactive follow-up
# -------------------------------------------------------------------------

print("\n=== INTERACTIVE FOLLOW-UP ===\n")

# Add a user question related to the refined idea
runner.step(user_utterance="What are the biggest technical hurdles to implementing your refined solid-state battery solution?")
print("Added user question to conversation")

# Get the system response (likely from the researcher)
response = runner.step()
print(f"\nResponse from {response.role}:")
print(response.utterance)

print("\nInteractive research session complete!") 