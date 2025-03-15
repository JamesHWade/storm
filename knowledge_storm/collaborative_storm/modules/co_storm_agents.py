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


class IdeaGeneration(dspy.Signature):
    """
    You are a creative research idea generator tasked with generating novel research ideas based on the conversation topic.
    Consider the current discussion and knowledge base to propose an innovative idea that pushes the boundaries of current thinking.
    The idea should be grounded in existing knowledge but explore new directions, combinations, or applications.
    
    Generate a research idea that is:
    1. Novel - introduces new concepts or approaches
    2. Feasible - could potentially be implemented
    3. Valuable - addresses important questions or challenges
    4. Related to the discussion topic
    
    Provide:
    - A clear description of the idea
    - Why it's novel or innovative
    - How it relates to the current discussion
    """
    
    topic = dspy.InputField(prefix="Topic of discussion:", format=str)
    current_knowledge = dspy.InputField(prefix="Current knowledge summary:", format=str)
    last_utterance = dspy.InputField(prefix="Last utterance in conversation:", format=str)
    idea = dspy.OutputField(format=str)


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
        self.idea_generator = dspy.Predict(IdeaGeneration)
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
    
    def generate_idea(self, topic: str, knowledge_summary: str, last_utterance: str):
        """Generate a novel research idea based on the conversation context."""
        try:
            with self.logging_wrapper.log_event("Researcher: generate idea"):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    idea = self.idea_generator(
                        topic=topic,
                        current_knowledge=knowledge_summary,
                        last_utterance=last_utterance
                    ).idea
        except RuntimeError as e:
            # Handle case where there's no active pipeline stage
            if "No pipeline stage is currently active" in str(e):
                with dspy.settings.context(lm=self.lm_config.discourse_manage_lm, show_guidelines=False):
                    idea = self.idea_generator(
                        topic=topic,
                        current_knowledge=knowledge_summary,
                        last_utterance=last_utterance
                    ).idea
            else:
                raise  # Re-raise if it's a different error
        return idea
    
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
                    # Generate idea
                    idea = self.generate_idea(
                        topic=self.topic,
                        knowledge_summary=conversation_summary,
                        last_utterance=last_utterance
                    )
                    
                    # Assess the idea
                    assessment = self.assess_idea(
                        idea=idea,
                        topic=self.topic
                    )
                    
                    # Create an experimental plan if appropriate
                    # Only create a plan sometimes to avoid too much detail in every response
                    include_plan = np.random.random() > 0.5
                    
                    if include_plan:
                        plan = self.create_experimental_plan(
                            idea=idea,
                            assessment=assessment
                        )
                        full_response = f"{idea}\n\n{assessment}\n\n{plan}"
                    else:
                        full_response = f"{idea}\n\n{assessment}"
                    
                    # Create conversation turn
                    conversation_turn = ConversationTurn(
                        role=self.role_name,
                        raw_utterance=full_response,
                        utterance_type="Further Details",
                        utterance=full_response
                    )
                    
                    return conversation_turn
