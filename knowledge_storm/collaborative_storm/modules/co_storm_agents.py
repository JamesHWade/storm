from itertools import zip_longest
from typing import TYPE_CHECKING, List, Optional

import dspy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ...dataclass import ConversationTurn, KnowledgeBase
from ...encoder import Encoder
from ...interface import Agent, Information, LMConfigs
from ...logging_wrapper import LoggingWrapper
from .callback import BaseCallbackHandler
from .collaborative_storm_utils import (
    _get_answer_question_module_instance,
    extract_storm_info_snippet,
)
from .costorm_expert_utterance_generator import CoStormExpertUtteranceGenerationModule
from .grounded_question_generation import GroundedQuestionGenerationModule
from .simulate_user import GenSimulatedUserUtterance

if TYPE_CHECKING:
    from ..engine import RunnerArgument


class CoStormExpert(Agent):
    """
    Represents an expert agent in the Co-STORM framework.
    The `CoStormExpert` is a specialized type of `Agent` that is tasked with participating in roundtable discussions within the Co-STORM system.
    The expert uses language models to generate action plans, answer questions, and polish its utterances based on the current conversation history and knowledge base.
      It interacts with modules for action planning and question answering grounding on provided retrieval models.

    Args:
        topic (str): The conversation topic that the expert specializes in.
        role_name (str): The perspective of the expert's role (e.g. AI enthusiast, drug discovery expert, etc.)
        role_description (str): A description of the perspective of the experts
        lm_config (LMConfigs): Configuration for the language models
        runner_argument (RunnerArgument): Co-STORM runner argument
        logging_wrapper (LoggingWrapper): An instance of `LoggingWrapper` to log events.
        rm (Optional[dspy.Retrieve], optional): A retrieval module used for fetching external knowledge or context.
        callback_handler (BaseCallbackHandler, optional): Handles log message printing
    """

    def __init__(
        self,
        topic: str,
        role_name: str,
        role_description: str,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        super().__init__(topic, role_name, role_description)
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler
        self.costorm_agent_utterance_generator = self._get_costorm_expert_utterance_generator(rm=rm)

    def _get_costorm_expert_utterance_generator(self, rm: Optional[dspy.Retrieve] = None):
        return CoStormExpertUtteranceGenerationModule(
            action_planning_lm=self.lm_config.discourse_manage_lm,
            utterance_polishing_lm=self.lm_config.utterance_polishing_lm,
            answer_question_module=_get_answer_question_module_instance(
                lm_config=self.lm_config,
                runner_argument=self.runner_argument,
                logging_wrapper=self.logging_wrapper,
                rm=rm,
            ),
            logging_wrapper=self.logging_wrapper,
            callback_handler=self.callback_handler,
        )

    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
    ):
        with self.logging_wrapper.log_event(
            "CoStormExpert generate utternace: get knowledge base summary"
        ):
            if self.callback_handler is not None:
                self.callback_handler.on_expert_action_planning_start()
            conversation_summary = knowledge_base.get_knowledge_base_summary()
        with self.logging_wrapper.log_event("CoStormExpert.generate_utterance generate utterance"):
            last_conv_turn = conversation_history[-1]
            conv_turn = self.costorm_agent_utterance_generator(
                topic=self.topic,
                current_expert=self.get_role_description(),
                conversation_summary=conversation_summary,
                last_conv_turn=last_conv_turn,
            ).conversation_turn
        with self.logging_wrapper.log_event("CoStormExpert generate utterance: polish utterance"):
            if self.callback_handler is not None:
                self.callback_handler.on_expert_utterance_polishing_start()
            self.costorm_agent_utterance_generator.polish_utterance(
                conversation_turn=conv_turn, last_conv_turn=last_conv_turn
            )
        return conv_turn


class SimulatedUser(Agent):
    """
    Simulated Users is a special type of Agent in Co-STORM that simulates real user interaction behavior based on the given intent.

    This class can be used for automatic experiments.
    For more information, please refer to Section 3.4 of Co-STORM paper: https://www.arxiv.org/pdf/2408.15232
    """

    def __init__(
        self,
        topic: str,
        role_name: str,
        role_description: str,
        intent: str,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        callback_handler: BaseCallbackHandler = None,
    ):
        super().__init__(topic, role_name, role_description)
        self.intent = intent
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.gen_simulated_user_utterance = GenSimulatedUserUtterance(
            engine=self.lm_config.question_answering_lm
        )
        self.callback_handler = callback_handler

    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
    ):
        assert self.intent is not None and self.intent, "Simulate user intent is not initialized."

        with self.logging_wrapper.log_event("SimulatedUser generate utternace: generate utterance"):
            utterance = self.gen_simulated_user_utterance(
                topic=self.topic, intent=self.intent, conv_history=conversation_history
            )
        return ConversationTurn(
            role="Guest", raw_utterance=utterance, utterance_type="Original Question"
        )


class Moderator(Agent):
    """
    The moderator's role in the Co-STORM framework is to inject new perspectives into the conversation to avoid stagnation, repetition, or overly niche discussions.
    This is achieved by generating questions based on unused, uncited snippets of information retrieved since the last moderator's turn.
    The selected information is reranked according to its relevance to the conversation topic and its dissimilarity to the original question.
    The resulting top-ranked snippets are used to generate an informed question to be presented to the conversation participants.

    For more information, please refer to Section 3.5 of Co-STORM paper: https://www.arxiv.org/pdf/2408.15232
    """

    def __init__(
        self,
        topic: str,
        role_name: str,
        role_description: str,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        encoder: Encoder,
        callback_handler: BaseCallbackHandler = None,
    ):
        super().__init__(topic, role_name, role_description)
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.grounded_question_generation_module = GroundedQuestionGenerationModule(
            engine=self.lm_config.question_asking_lm
        )
        self.callback_handler = callback_handler
        self.encoder = encoder

    def _get_conv_turn_unused_information(
        self, conv_turn: ConversationTurn, knowledge_base: KnowledgeBase
    ):
        # extract all snippets from raw retrieved information
        raw_retrieved_info: List[Information] = conv_turn.raw_retrieved_info
        raw_retrieved_single_snippet_info: List[Information] = []
        for info in raw_retrieved_info:
            for snippet_idx in range(len(info.snippets)):
                raw_retrieved_single_snippet_info.append(
                    extract_storm_info_snippet(info, snippet_index=snippet_idx)
                )
        # get all cited information
        cited_info = list(knowledge_base.info_uuid_to_info_dict.values())
        cited_info_hash_set = set([hash(info) for info in cited_info])
        cited_snippets = [info.snippets[0] for info in cited_info]
        # get list of unused information
        unused_information: List[Information] = [
            info
            for info in raw_retrieved_single_snippet_info
            if hash(info) not in cited_info_hash_set
        ]
        if not unused_information:
            return []
        # extract snippets to get embeddings
        unused_information_snippets = [info.snippets[0] for info in unused_information]
        # get embeddings
        unused_snippets_embeddings = self.encoder.encode(
            unused_information_snippets, max_workers=20
        )
        claim_embedding = self.encoder.encode(conv_turn.claim_to_make)
        query_embedding = self.encoder.encode(conv_turn.queries)
        cited_snippets_embedding = self.encoder.encode(cited_snippets)
        # calculate similarity
        query_similarities = cosine_similarity(unused_snippets_embeddings, query_embedding)
        max_query_similarity = np.max(query_similarities, axis=1)
        cited_snippets_similarity = np.max(
            cosine_similarity(unused_snippets_embeddings, cited_snippets_embedding),
            axis=1,
        )
        cited_snippets_similarity = np.clip(cited_snippets_similarity, 0, 1)
        # use claim similarity to filter out "real" not useful data
        claim_similarity = cosine_similarity(
            unused_snippets_embeddings, claim_embedding.reshape(1, -1)
        ).flatten()
        claim_similarity = np.where(claim_similarity >= 0.25, 1.0, 0.0)
        # calculate score: snippet that is close to topic but far from query
        query_sim_weight = 0.5
        cited_snippets_sim_weight = 1 - query_sim_weight
        combined_scores = (
            ((1 - max_query_similarity) ** query_sim_weight)
            * ((1 - cited_snippets_similarity) ** cited_snippets_sim_weight)
            * claim_similarity
        )
        sorted_indices = np.argsort(combined_scores)[::-1]
        return [unused_information[idx] for idx in sorted_indices]

    def _get_sorted_unused_snippets(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
        last_n_conv_turn: int = 2,
    ):
        # get last N conv turn and batch encode all related strings
        considered_conv_turn = []
        batch_snippets = [self.topic]
        for conv_turn in reversed(conversation_history):
            if len(considered_conv_turn) == last_n_conv_turn:
                break
            if conv_turn.utterance_type == "Questioning":
                break
            considered_conv_turn.append(conv_turn)
            batch_snippets.extend(sum([info.snippets for info in conv_turn.raw_retrieved_info], []))
            batch_snippets.append(conv_turn.claim_to_make)
            batch_snippets.extend(conv_turn.queries)
        self.encoder.encode(batch_snippets, max_workers=20)

        # get sorted unused snippets for each turn
        sorted_snippets = []
        for conv_turn in considered_conv_turn:
            sorted_snippets.append(
                self._get_conv_turn_unused_information(
                    conv_turn=conv_turn, knowledge_base=knowledge_base
                )
            )

        # use round robin rule to merge these snippets
        merged_snippets = []
        for elements in zip_longest(*sorted_snippets, fillvalue=None):
            merged_snippets.extend(e for e in elements if e is not None)
        return merged_snippets

    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
    ):
        with self.logging_wrapper.log_event("Moderator generate utternace: get unused snippets"):
            unused_snippets: List[Information] = self._get_sorted_unused_snippets(
                knowledge_base=knowledge_base, conversation_history=conversation_history
            )
        with self.logging_wrapper.log_event(
            "Moderator generate utternace: QuestionGeneration module"
        ):
            generated_question = self.grounded_question_generation_module(
                topic=self.topic,
                knowledge_base=knowledge_base,
                last_conv_turn=conversation_history[-1],
                unused_snippets=unused_snippets,
            )
        return ConversationTurn(
            role=self.role_name,
            raw_utterance=generated_question.raw_utterance,
            utterance_type="Original Question",
            utterance=generated_question.utterance,
            cited_info=generated_question.cited_info,
        )


class PureRAGAgent(Agent):
    """
    PureRAGAgent only handles grounded question generation by retrieving information from the retriever based on the query.
    It does not utilize any other information besides the query itself.

    It's designed for Co-STORM paper baseline comparison.
    """

    def __init__(
        self,
        topic: str,
        role_name: str,
        role_description: str,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        super().__init__(topic, role_name, role_description)
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.grounded_question_answering_module = _get_answer_question_module_instance(
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=rm,
        )

    def _gen_utterance_from_question(self, question: str):
        grounded_answer = self.grounded_question_answering_module(
            topic=self.topic,
            question=question,
            mode="brief",
            style="conversational and concise",
        )
        conversation_turn = ConversationTurn(
            role=self.role_name, raw_utterance="", utterance_type="Potential Answer"
        )
        conversation_turn.claim_to_make = question
        conversation_turn.raw_utterance = grounded_answer.response
        conversation_turn.utterance = grounded_answer.response
        conversation_turn.queries = grounded_answer.queries
        conversation_turn.raw_retrieved_info = grounded_answer.raw_retrieved_info
        conversation_turn.cited_info = grounded_answer.cited_info
        return conversation_turn

    def generate_topic_background(self):
        return self._gen_utterance_from_question(self.topic)

    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
    ):
        with self.logging_wrapper.log_event("PureRAGAgent generate utternace: generate utterance"):
            return self._gen_utterance_from_question(question=conversation_history[-1].utterance)

class IdeaCreation(dspy.Signature):
    """
    You are a creative research idea generator tasked with producing a diverse set of potential research ideas.
    
    Consider the topic, current knowledge, and conversation context to generate a variety of potential research directions.
    Each idea should be expressed as a single sentence that captures its core essence.
    
    Your ideas should:
    1. Be diverse - explore different approaches, methodologies, and perspectives
    2. Be novel - introduce new concepts or approaches
    3. Be feasible - have potential for implementation
    4. Be valuable - address important questions or challenges
    5. Be relevant - connect to the discussion topic
    
    Create distinct ideas, with each representing a different approach or angle.
    Number each idea and keep each to a single sentence that captures the core concept.
    """
    
    topic = dspy.InputField(prefix="Topic of discussion:", format=str)
    current_knowledge = dspy.InputField(prefix="Current knowledge summary:", format=str)
    conversation_context = dspy.InputField(prefix="Recent conversation context:", format=str)
    idea_list = dspy.OutputField(prefix="List of brief research ideas:", format=list)


class IdeaCuration(dspy.Signature):
    """
    You are a research idea curator tasked with selecting and developing the most promising research idea from a list.
    
    Evaluate each of the provided ideas based on:
    1. Novelty - how original and innovative is the idea?
    2. Feasibility - how practical is it to implement?
    3. Value - what impact could this idea have?
    4. Relevance - how well does it address the topic and context?
    
    First, analyze each idea briefly, noting its strengths and weaknesses.
    Then, select the most promising idea and develop it into a comprehensive research concept that includes:
    - A thorough description of the concept and approach
    - The specific innovation or novelty that distinguishes this idea
    - The potential value and impact of pursuing this research
    - How it builds upon or challenges existing knowledge
    - Why this idea is particularly relevant to the broader topic
    
    Clearly indicate which idea number you've selected and why, then develop it fully.
    """
    
    topic = dspy.InputField(prefix="Topic of discussion:", format=str)
    idea_list = dspy.InputField(prefix="List of brief research ideas:", format=list)
    current_knowledge = dspy.InputField(prefix="Current knowledge summary:", format=str)
    conversation_context = dspy.InputField(prefix="Recent conversation context:", format=str)
    selected_idea_number = dspy.OutputField(prefix="Selected idea number:", format=str)
    selection_reasoning = dspy.OutputField(prefix="Reasoning for selection:", format=str)
    developed_idea = dspy.OutputField(prefix="Fully developed research idea:", format=str)


class IdeaRefinement(dspy.Signature):
    """
    You are a research idea refinement specialist. Your task is to improve an existing research idea based on feedback.
    
    Consider:
    1. The original research idea's strengths and core concepts
    2. The specific feedback provided
    3. How to address limitations or concerns while preserving the innovative aspects
    4. Ways to make the idea more feasible, valuable, or novel as suggested by the feedback
    
    Provide a refined version of the research idea that:
    - Maintains the original core concept
    - Addresses the feedback points
    - Improves overall quality and potential impact
    - Is more detailed and well-developed than the original
    """
    
    original_idea = dspy.InputField(prefix="Original research idea:", format=str)
    feedback = dspy.InputField(prefix="Feedback on the idea:", format=str)
    topic = dspy.InputField(prefix="Research topic:", format=str)
    refined_idea = dspy.OutputField(format=str)


class IdeaAssessment(dspy.Signature):
    """
    Assess the proposed research idea considering its:
    1. Novelty - how original is this idea?
    2. Feasibility - how practical would it be to implement?
    3. Value - what benefits might result if successful?
    4. Relevance - how well does it relate to the discussion topic?
    
    Provide a balanced assessment addressing strengths and limitations.
    Score each aspect from 1-10 (10 being highest) and provide a brief justification.
    """
    
    idea = dspy.InputField(prefix="Research idea to assess:", format=str)
    topic = dspy.InputField(prefix="Discussion topic:", format=str)
    assessment = dspy.OutputField(format=str)


class ExperimentalPlan(dspy.Signature):
    """
    Create a detailed experimental plan to test or validate the proposed research idea.
    Consider:
    1. Methodology - what approach would be most appropriate?
    2. Required resources - what tools, data, or expertise would be needed?
    3. Potential challenges - what obstacles might arise and how could they be addressed?
    4. Timeline - rough estimate of how long implementation might take
    5. Expected outcomes - what results might validate or refute the idea?
    
    The plan should be realistic but ambitious, with clear steps for implementation.
    """
    
    idea = dspy.InputField(prefix="Research idea to test:", format=str)
    idea_assessment = dspy.InputField(prefix="Assessment of idea:", format=str)
    plan = dspy.OutputField(format=str)


class Researcher(Agent):
    """
    The Researcher agent in the Co-STORM framework specializes in generating novel ideas, assessing them for aspects 
    like novelty, feasibility, and value, and proposing experimental plans to test these ideas.
    
    This agent brings creative thinking and scientific rigor to conversations, helping to push the boundaries
    of current knowledge while providing practical pathways to test new concepts. The Researcher uses language
    models to generate ideas based on the conversation context, assess their merits, and develop experimental
    plans grounded in scientific methodology.
    
    Args:
        topic (str): The conversation topic that the researcher focuses on.
        role_name (str): The specific researcher role (e.g., "Innovation Scientist", "Creative Researcher").
        role_description (str): A description of the researcher's perspective and approach.
        lm_config (LMConfigs): Configuration for the language models.
        runner_argument (RunnerArgument): Co-STORM runner argument.
        logging_wrapper (LoggingWrapper): An instance of LoggingWrapper to log events.
        rm (Optional[dspy.Retrieve], optional): A retrieval module for fetching relevant information.
        callback_handler (BaseCallbackHandler, optional): Handles log message printing.
    """
    
    def __init__(
        self,
        topic: str,
        role_name: str,
        role_description: str,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        super().__init__(topic, role_name, role_description)
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler
        
        # Initialize specialized modules for the researcher
        self.idea_creator = dspy.Predict(IdeaCreation)
        self.idea_curator = dspy.Predict(IdeaCuration)
        self.idea_assessor = dspy.Predict(IdeaAssessment)
        self.experiment_planner = dspy.Predict(ExperimentalPlan)
        self.idea_refiner = dspy.Predict(IdeaRefinement)
        
        # For grounded responses
        self.grounded_question_answering_module = _get_answer_question_module_instance(
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=rm,
        )
    
    def generate_idea(self, topic: str, knowledge_summary: str, conversation_context: str):
        """
        Generate a novel research idea using a two-stage process of creation and curation.
        
        This method first generates multiple brief idea candidates, then selects and develops
        the most promising one. The process leverages comprehensive conversation context 
        and knowledge to ensure ideas are relevant and well-grounded.
        
        Args:
            topic (str): The research topic to focus on
            knowledge_summary (str): A summary of relevant knowledge on the topic
            conversation_context (str): Detailed context from recent conversation history
            
        Returns:
            str: A formatted output containing idea candidates, selection reasoning, and the developed idea
        """
        try:
            with self.logging_wrapper.log_event("Researcher: generate multiple idea candidates"):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    # Stage 1: Generate multiple brief ideas
                    idea_creation_result = self.idea_creator(
                        topic=topic,
                        current_knowledge=knowledge_summary,
                        conversation_context=conversation_context
                    )
                    
                    # Ensure idea_list is properly handled as a list of distinct ideas
                    # If it's a string, split it into lines and clean it up
                    idea_list = idea_creation_result.idea_list
                    if isinstance(idea_list, str):
                        # Split by newlines and filter out empty lines
                        idea_list = [line.strip() for line in idea_list.split('\n') if line.strip()]
                        # If we still don't have proper ideas, try to extract numbered ideas
                        if len(idea_list) <= 1:
                            # Try to extract numbered items (e.g., "1. First idea")
                            import re
                            matches = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', idea_list[0], re.DOTALL)
                            if matches:
                                idea_list = [match.strip() for match in matches if match.strip()]
                    
                    # Ensure we have at least one idea
                    if not idea_list or (len(idea_list) == 1 and not idea_list[0]):
                        idea_list = [f"A novel approach to {topic} that builds on existing research"]
                    
                    # Remove any existing numbering from the ideas (e.g., "1. ", "2. ")
                    import re
                    cleaned_idea_list = []
                    for idea in idea_list:
                        # Remove numbering patterns like "1.", "1)", "#1" at the beginning of the idea
                        cleaned_idea = re.sub(r'^(\d+\.|\d+\)|\#\d+|\d+)\s*', '', idea).strip()
                        cleaned_idea_list.append(cleaned_idea)
                    
                    # Format idea list for display
                    formatted_idea_list = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(cleaned_idea_list)])
                    
                    # Stage 2: Select and develop the best idea
                    idea_curation_result = self.idea_curator(
                        topic=topic,
                        idea_list=cleaned_idea_list,  # Use the cleaned list for curation
                        current_knowledge=knowledge_summary,
                        conversation_context=conversation_context
                    )
                    
                    # Get the selected idea index, ensuring it's valid
                    try:
                        selected_idx = int(idea_curation_result.selected_idea_number) - 1
                        if selected_idx < 0 or selected_idx >= len(cleaned_idea_list):
                            selected_idx = 0
                    except (ValueError, TypeError):
                        selected_idx = 0
                    
                    # Format the final output to include both the candidates and selected idea
                    final_idea = f"""# Research Idea Candidates
{formatted_idea_list}

# Selected Idea: {cleaned_idea_list[selected_idx]}
## Selection Rationale:
{idea_curation_result.selection_reasoning}

## Developed Research Idea:
{idea_curation_result.developed_idea}
"""
        except RuntimeError as e:
            # Handle case where there's no active pipeline stage
            if "No pipeline stage is currently active" in str(e):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    # Create a simple fallback that still generates multiple ideas
                    # but in a more streamlined way
                    try:
                        idea_creation = self.idea_creator(
                            topic=topic,
                            current_knowledge=knowledge_summary,
                            conversation_context=conversation_context
                        )
                        fallback_ideas = idea_creation.idea_list
                        
                        # Process fallback ideas the same way
                        if isinstance(fallback_ideas, str):
                            # Split by newlines and filter out empty lines
                            fallback_ideas = [line.strip() for line in fallback_ideas.split('\n') if line.strip()]
                            # Check if we need to extract numbered ideas
                            if len(fallback_ideas) <= 1 and fallback_ideas:
                                import re
                                matches = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', fallback_ideas[0], re.DOTALL)
                                if matches:
                                    fallback_ideas = [match.strip() for match in matches if match.strip()]
                        
                        # Remove any existing numbering from the fallback ideas
                        cleaned_fallback_ideas = []
                        for idea in fallback_ideas:
                            # Remove numbering patterns
                            cleaned_idea = re.sub(r'^(\d+\.|\d+\)|\#\d+|\d+)\s*', '', idea).strip()
                            cleaned_fallback_ideas.append(cleaned_idea)
                        
                        # Ensure we have at least one idea
                        if cleaned_fallback_ideas and len(cleaned_fallback_ideas) > 0:
                            first_idea = cleaned_fallback_ideas[0]
                            final_idea = f"# Research Idea\n{first_idea}\n\nUnable to perform full curation process."
                        else:
                            # Ultimate fallback
                            final_idea = f"# Research Idea\nA novel approach to {topic} that builds on existing research."
                    except:
                        # Ultimate fallback if everything fails
                        final_idea = f"# Research Idea\nA novel approach to {topic} that builds on existing research."
            else:
                raise  # Re-raise if it's a different error
        return final_idea
    
    def assess_idea(self, idea: str, topic: str):
        """Assess the generated idea for novelty, feasibility, value, and relevance."""
        try:
            with self.logging_wrapper.log_event("Researcher: assess idea"):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    assessment = self.idea_assessor(
                        idea=idea,
                        topic=topic
                    ).assessment
        except RuntimeError as e:
            # Handle case where there's no active pipeline stage
            if "No pipeline stage is currently active" in str(e):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    assessment = self.idea_assessor(
                        idea=idea,
                        topic=topic
                    ).assessment
            else:
                raise  # Re-raise if it's a different error
        return assessment
    
    def create_experimental_plan(self, idea: str, assessment: str):
        """Create an experimental plan to test the research idea."""
        try:
            with self.logging_wrapper.log_event("Researcher: create experimental plan"):
                with dspy.settings.context(lm=self.lm_config.utterance_polishing_lm, show_guidelines=False):
                    plan = self.experiment_planner(
                        idea=idea,
                        idea_assessment=assessment
                    ).plan
        except RuntimeError as e:
            # Handle case where there's no active pipeline stage
            if "No pipeline stage is currently active" in str(e):
                with dspy.settings.context(lm=self.lm_config.utterance_polishing_lm, show_guidelines=False):
                    plan = self.experiment_planner(
                        idea=idea,
                        idea_assessment=assessment
                    ).plan
            else:
                raise  # Re-raise if it's a different error
        return plan
    
    def refine_idea(self, original_idea: str, feedback: str, topic: str = None):
        """Refine a research idea based on feedback.
        
        Args:
            original_idea (str): The original research idea to refine
            feedback (str): Feedback about the idea, including criticisms, suggestions, or questions
            topic (str, optional): The research topic. Defaults to self.topic if None.
            
        Returns:
            str: A refined version of the research idea that addresses the feedback
        """
        if topic is None:
            topic = self.topic
            
        try:
            with self.logging_wrapper.log_event("Researcher: refine idea"):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    refined_idea = self.idea_refiner(
                        original_idea=original_idea,
                        feedback=feedback,
                        topic=topic
                    ).refined_idea
        except RuntimeError as e:
            # Handle case where there's no active pipeline stage
            if "No pipeline stage is currently active" in str(e):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    refined_idea = self.idea_refiner(
                        original_idea=original_idea,
                        feedback=feedback,
                        topic=topic
                    ).refined_idea
            else:
                raise  # Re-raise if it's a different error
        return refined_idea
    
    def generate_utterance(
        self,
        knowledge_base: KnowledgeBase,
        conversation_history: List[ConversationTurn],
    ):
        with self.logging_wrapper.log_event("Researcher generate utterance: processing context"):
            conversation_summary = knowledge_base.get_knowledge_base_summary()
            last_conv_turn = conversation_history[-1]
            last_utterance = last_conv_turn.utterance
            
            # Determine appropriate response type based on conversation flow
            if last_conv_turn.utterance_type in ["Original Question", "Information Request"]:
                # If someone asked a direct question, provide a grounded answer
                with self.logging_wrapper.log_event("Researcher: answering direct question"):
                    grounded_answer = self.grounded_question_answering_module(
                        topic=self.topic,
                        question=last_utterance,
                        mode="brief",
                        style="conversational and detailed",
                    )
                    
                    conversation_turn = ConversationTurn(
                        role=self.role_name, 
                        raw_utterance="", 
                        utterance_type="Potential Answer"
                    )
                    conversation_turn.claim_to_make = last_utterance
                    conversation_turn.raw_utterance = grounded_answer.response
                    conversation_turn.utterance = grounded_answer.response
                    conversation_turn.queries = grounded_answer.queries
                    conversation_turn.raw_retrieved_info = grounded_answer.raw_retrieved_info
                    conversation_turn.cited_info = grounded_answer.cited_info
                    
                    return conversation_turn
            else:
                # Generate a novel idea and assessment
                with self.logging_wrapper.log_event("Researcher: generating novel content"):
                    # Prepare rich conversation context
                    conversation_context = ""
                    if conversation_history:
                        # Include up to 10 recent turns for rich context
                        recent_turns = conversation_history[-10:]
                        conversation_context = "Recent Conversation Context:\n"
                        for i, turn in enumerate(recent_turns):
                            role = turn.role
                            utterance = turn.utterance
                            utterance_type = turn.utterance_type if hasattr(turn, 'utterance_type') else "Message"
                            conversation_context += f"Turn {len(conversation_history) - len(recent_turns) + i + 1} - {role} ({utterance_type}):\n{utterance}\n\n"
                    
                    # Add immediate context
                    last_turn_context = f"Most Recent Message: {last_conv_turn.role} - {last_utterance}"
                    full_context = f"{conversation_context}\n{last_turn_context}"
                    
                    # Generate idea with rich context
                    idea_output = self.generate_idea(
                        topic=self.topic,
                        knowledge_summary=conversation_summary,
                        conversation_context=full_context
                    )
                    
                    # The idea_output now contains multiple sections with candidates, selection, and developed idea
                    # For the conversation, we'll focus primarily on the developed idea
                    # but can reference the selection process as well
                    
                    # Extract the developed idea from the output - it's in the final section
                    idea_parts = idea_output.split("## Developed Research Idea:")
                    if len(idea_parts) > 1:
                        developed_idea = idea_parts[1].strip()
                        selection_parts = idea_output.split("## Selection Rationale:")
                        selection_rationale = ""
                        if len(selection_parts) > 1:
                            selection_rationale = selection_parts[1].split("## Developed Research Idea:")[0].strip()
                        
                        # Structure the idea for presentation in conversation
                        idea_for_assessment = f"# Research Idea\n{developed_idea}"
                        
                        # Include a condensed version of the selection rationale if available
                        if selection_rationale:
                            idea_for_assessment = f"# Selection Rationale\n{selection_rationale}\n\n{idea_for_assessment}"
                    else:
                        # Fallback if parsing fails
                        idea_for_assessment = idea_output
                    
                    # Assess the idea
                    assessment = self.assess_idea(
                        idea=idea_for_assessment,
                        topic=self.topic
                    )
                    
                    # Create an experimental plan if appropriate
                    # Only create a plan sometimes to avoid too much detail in every response
                    include_plan = np.random.random() > 0.5
                    
                    if include_plan:
                        plan = self.create_experimental_plan(
                            idea=idea_for_assessment,
                            assessment=assessment
                        )
                        full_response = f"{idea_output}\n\n{assessment}\n\n{plan}"
                    else:
                        full_response = f"{idea_output}\n\n{assessment}"
                    
                    # Create conversation turn
                    conversation_turn = ConversationTurn(
                        role=self.role_name,
                        raw_utterance=full_response,
                        utterance_type="Further Details",
                        utterance=full_response
                    )
                    
                    return conversation_turn
