import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Union

import dspy

from ..dataclass import ConversationTurn, KnowledgeBase
from ..encoder import Encoder
from ..interface import Agent, LMConfigs
from ..logging_wrapper import LoggingWrapper
from ..rm import BingSearch
from .modules import collaborative_storm_utils as collaborative_storm_utils
from .modules.callback import BaseCallbackHandler
from .modules.co_storm_agents import (
    CoStormExpert,
    Moderator,
    PureRAGAgent,
    Researcher,
    SimulatedUser,
)
from .modules.expert_generation import GenerateExpertModule
from .modules.warmstart_hierarchical_chat import WarmStartModule


class CollaborativeStormLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of Co-STORM.

    Given that different parts in Co-STORM framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        self.question_answering_lm = None
        self.discourse_manage_lm = None
        self.utterance_polishing_lm = None
        self.warmstart_outline_gen_lm = None
        self.question_asking_lm = None
        self.knowledge_base_lm = None

    def init(
        self,
        lm_type: Literal["openai", "azure", "together"],
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        if lm_type and lm_type == "openai":
            openai_kwargs = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
                "api_base": None,
            }
            self.question_answering_lm = dspy.LM(
                model="gpt-4o-2024-11-20", max_tokens=4000, **openai_kwargs
            )
            self.discourse_manage_lm = dspy.LM(
                model="gpt-4o-2024-11-20", max_tokens=4000, **openai_kwargs
            )
            self.utterance_polishing_lm = dspy.LM(
                model="gpt-4o-2024-11-20", max_tokens=4000, **openai_kwargs
            )
            self.warmstart_outline_gen_lm = dspy.LM(
                model="gpt-4-1106-preview", max_tokens=4000, **openai_kwargs
            )
            self.question_asking_lm = dspy.LM(
                model="gpt-4o-2024-11-20", max_tokens=3000, **openai_kwargs
            )
            self.knowledge_base_lm = dspy.LM(
                model="gpt-4o-2024-11-20", max_tokens=4000, **openai_kwargs
            )
        elif lm_type and lm_type == "azure":
            azure_kwargs = {
                "api_key": os.getenv("AZURE_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            }
            self.question_answering_lm = dspy.LM(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs
            )
            self.discourse_manage_lm = dspy.LM(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs
            )
            self.utterance_polishing_lm = dspy.LM(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs
            )
            self.warmstart_outline_gen_lm = dspy.LM(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs
            )
            self.question_asking_lm = dspy.LM(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs
            )
            self.knowledge_base_lm = dspy.LM(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs
            )

    def set_question_answering_lm(self, model: dspy.LM):
        self.question_answering_lm = model

    def set_discourse_manage_lm(self, model: dspy.LM):
        self.discourse_manage_lm = model

    def set_utterance_polishing_lm(self, model: dspy.LM):
        self.utterance_polishing_lm = model

    def set_warmstart_outline_gen_lm(self, model: dspy.LM):
        self.warmstart_outline_gen_lm = model

    def set_question_asking_lm(self, model: dspy.LM):
        self.question_asking_lm = model

    def set_knowledge_base_lm(self, model: dspy.LM):
        self.knowledge_base_lm = model

    def collect_and_reset_lm_usage(self):
        lm_usage = {}
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(
                getattr(self, attr_name), "get_usage_and_reset"
            ):
                usage = getattr(self, attr_name).get_usage_and_reset()
                if any(
                    value["prompt_tokens"] != 0 or value["completion_tokens"] != 0
                    for value in usage.values()
                ):
                    lm_usage[attr_name] = usage
        return lm_usage

    def to_dict(self):
        """
        Converts the CollaborativeStormLMConfigs instance to a dictionary representation.
        Sanitizes all sensitive information such as API keys, tokens, and credentials.
        """
        config_dict = {}

        # For each LM attribute
        for attr_name, attr_value in self.__dict__.items():
            if attr_name.endswith("_lm") and attr_value is not None:
                # Store the model information directly
                model_info = {
                    "type": attr_value.__class__.__name__,
                    "module": attr_value.__class__.__module__,
                    "model": getattr(attr_value, "model", None),
                    "max_tokens": getattr(attr_value, "max_tokens", None),
                }

                # Check for additional important attributes
                for param in ["temperature", "top_p"]:
                    if hasattr(attr_value, param):
                        model_info[param] = getattr(attr_value, param)

                # Check if the model has a kwargs dictionary
                if hasattr(attr_value, "kwargs"):
                    # Make a sanitized copy to avoid reference issues and remove sensitive data
                    sanitized_kwargs = {}

                    # Only copy non-sensitive keys
                    sensitive_keys = [
                        "api_key",
                        "API_KEY",
                        "apikey",
                        "api-key",
                        "token",
                        "auth",
                        "authorization",
                        "auth_token",
                        "bearer",
                        "secret",
                        "password",
                        "credential",
                        "client_secret",
                        "client_id",
                        "tenant_id",
                    ]

                    for k, v in attr_value.kwargs.items():
                        # Skip any key that appears to contain sensitive information
                        if any(
                            sensitive_term in k.lower()
                            for sensitive_term in sensitive_keys
                        ):
                            continue

                        # Handle extra_headers specially
                        if k == "extra_headers":
                            sanitized_kwargs[k] = "AUTH_HEADERS_PRESENT"
                        # Handle API base URLs - keep them but indicate they were present
                        elif k in ["api_base", "endpoint", "base_url"]:
                            if v is not None:
                                sanitized_kwargs[k] = "API_BASE_PRESENT"
                        # Include other non-sensitive values
                        else:
                            sanitized_kwargs[k] = v

                    # Only add kwargs to model_info if there are any non-sensitive keys
                    if sanitized_kwargs:
                        model_info["kwargs"] = sanitized_kwargs

                    # Add a flag to indicate this model used authentication
                    if (
                        "api_key" in attr_value.kwargs
                        or "extra_headers" in attr_value.kwargs
                    ):
                        model_info["used_authentication"] = True

                config_dict[attr_name] = model_info

        return config_dict

    @classmethod
    def from_dict(cls, data):
        """
        Constructs a CollaborativeStormLMConfigs instance from a dictionary representation.
        Restores sensitive information from environment variables instead of expecting it in the data.
        Properly handles both direct API key and header-based authentication.
        """
        # If there's no data, raise an exception
        if not data:
            raise ValueError(
                "No data provided to construct CollaborativeStormLMConfigs"
            )

        # Create a new instance
        config = cls()

        # Import LM class
        from dspy import LM

        for attr_name, model_info in data.items():
            if attr_name.endswith("_lm") and model_info:
                # Extract model class if specified
                LMClass = LM
                if "module" in model_info and "type" in model_info:
                    try:
                        module_name = model_info["module"]
                        class_name = model_info["type"]
                        module = __import__(module_name, fromlist=[class_name])
                        LMClass = getattr(module, class_name)
                    except (ImportError, AttributeError):
                        pass

                # Extract parameters from model_info
                model = model_info.get("model")
                kwargs = model_info.get("kwargs", {}).copy()
                max_tokens = model_info.get("max_tokens")
                temperature = model_info.get("temperature")
                top_p = model_info.get("top_p")

                # Add temperature and top_p to kwargs if present
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if top_p is not None:
                    kwargs["top_p"] = top_p

                # Prioritize restoring authentication - first determine if this is Azure
                is_azure = False
                if model and "azure" in str(model).lower():
                    is_azure = True
                elif (
                    "api_base" in kwargs
                    and kwargs["api_base"]
                    and "azure" in str(kwargs["api_base"]).lower()
                ):
                    is_azure = True

                # Check if we should use header-based auth (for DOW environment)
                client_id = os.getenv("AZURE_OPENAI_CLIENT_ID") or os.getenv(
                    "AZURE_CLIENT_ID"
                )
                client_secret = os.getenv("AZURE_OPENAI_CLIENT_SECRET") or os.getenv(
                    "AZURE_CLIENT_SECRET"
                )
                tenant_id = os.getenv("AZURE_OPENAI_TENANT_ID") or os.getenv(
                    "AZURE_TENANT_ID"
                )

                # Determine authentication method based on environment variables
                # Prioritize header-based auth if client credentials are available
                if is_azure and all([client_id, client_secret, tenant_id]):
                    # Use header-based authentication
                    auth_headers = get_auth_headers()
                    if auth_headers:
                        kwargs["extra_headers"] = auth_headers
                        kwargs["api_key"] = (
                            None  # Set explicitly to None for header auth
                        )
                elif is_azure:
                    # Fall back to API key for Azure
                    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv(
                        "AZURE_API_KEY"
                    )
                    if api_key:
                        kwargs["api_key"] = api_key
                else:
                    # Standard OpenAI authentication
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        kwargs["api_key"] = api_key

                # Add API base if it was sanitized or not present
                if is_azure and (
                    "api_base" not in kwargs
                    or kwargs.get("api_base") == "API_BASE_PRESENT"
                ):
                    api_base = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv(
                        "AZURE_API_BASE"
                    )
                    if api_base:
                        kwargs["api_base"] = api_base

                # Add API version for Azure if not present
                if is_azure and "api_version" not in kwargs:
                    api_version = os.getenv("AZURE_API_VERSION")
                    if api_version:
                        kwargs["api_version"] = api_version

                # Remove placeholders that were used to indicate presence
                if kwargs.get("extra_headers") == "AUTH_HEADERS_PRESENT":
                    del kwargs["extra_headers"]  # Remove placeholder

                if kwargs.get("api_base") == "API_BASE_PRESENT":
                    del kwargs["api_base"]  # Remove placeholder

                # Create the LM instance
                try:
                    lm = LMClass(
                        model=model,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    setattr(config, attr_name, lm)
                except Exception as e:
                    print(
                        f"Warning: Could not create {attr_name} with model {model}: {e}"
                    )

        return config


@dataclass
class RunnerArgument:
    """Arguments for controlling the STORM Wiki pipeline."""

    topic: str = field(
        metadata={"help": "Topic of discourse"},
    )
    retrieve_top_k: int = field(
        default=10,
        metadata={"help": "retrieve top k results for each query in retriever"},
    )
    max_search_queries: int = field(
        default=2,
        metadata={
            "help": "Maximum number of search queries to consider for each question."
        },
    )
    total_conv_turn: int = field(
        default=20,
        metadata={"help": "Maximum number turn in conversation."},
    )
    max_search_thread: int = field(
        default=5,
        metadata={"help": "Maximum number of parallel thread for retriever"},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    warmstart_max_num_experts: int = field(
        default=3,
        metadata={
            "help": "Max number of experts in perspective guided QA in warm start process"
        },
    )
    warmstart_max_turn_per_experts: int = field(
        default=2,
        metadata={"help": "Max number of turns per perspective in warm start process"},
    )
    warmstart_max_thread: int = field(
        default=3,
        metadata={
            "help": "Max number thread for parallel perspective guided QA in warm start process"
        },
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )
    max_num_round_table_experts: int = field(
        default=2,
        metadata={"help": "Max number of active experts in round table discussion."},
    )
    moderator_override_N_consecutive_answering_turn: int = field(
        default=3,
        metadata={
            "help": "Number of consecutive experts answering turn before moderator override the conversation"
        },
    )
    node_expansion_trigger_count: int = field(
        default=10,
        metadata={
            "help": "Trigger node expansion for node that contain more than N snippets"
        },
    )
    disable_moderator: bool = field(
        default=False,
        metadata={"help": "If True, disable moderator."},
    )
    disable_multi_experts: bool = field(
        default=False,
        metadata={"help": "If True, disable moderator."},
    )
    rag_only_baseline_mode: bool = field(
        default=False,
        metadata={"help": "If True, switch to rag online baseline mode"},
    )

    def to_dict(self):
        """
        Converts the RunnerArgument instance to a dictionary representation.

        Returns:
            dict: The dictionary representation of the RunnerArgument.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """
        Constructs a RunnerArgument instance from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the RunnerArgument.

        Returns:
            RunnerArgument: The constructed RunnerArgument instance.
        """
        return cls(**data)


@dataclass
class TurnPolicySpec:
    """
    Represents the policy specifications for determining the behavior of a conversation turn.

    Attributes:
        should_reorganize_knowledge_base (bool):
            A flag that indicates whether the knowledge base should be reorganized after the current turn.

        should_update_experts_list (bool):
            A flag that indicates whether the list of experts should be updated based on the conversation context.

        should_polish_utterance (bool):
            A flag that indicates whether the generated utterance should be polished (e.g., refined or rephrased) before it is used in the conversation.

        agent (Agent):
            The `Agent` responsible for generating utterances or responses during the conversation turn.
            This agent interacts with the knowledge base and the conversation history to produce responses.
    """

    should_reorganize_knowledge_base: bool = False
    should_update_experts_list: bool = False
    should_polish_utterance: bool = False
    agent: Agent = None


class DiscourseManager:
    def __init__(
        self,
        topic: str,
        runner_argument: RunnerArgument,
        lm_config: CollaborativeStormLMConfigs,
        encoder: Encoder,
        knowledge_base: KnowledgeBase,
        logging_wrapper: LoggingWrapper,
        rm: dspy.Retrieve,
        callback_handler: BaseCallbackHandler = None,
    ):
        # parameter management
        self.topic = topic
        self.lm_config = lm_config
        self.runner_argument = runner_argument
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler
        self.rm = rm
        self.encoder = encoder
        self.knowledge_base = knowledge_base
        # role management
        self.experts: List[CoStormExpert] = []
        self.simulated_user: SimulatedUser = SimulatedUser(
            topic=self.runner_argument.topic,
            role_name="Guest",
            role_description="",
            intent=None,
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            callback_handler=self.callback_handler,
        )
        self.pure_rag_agent: PureRAGAgent = PureRAGAgent(
            topic=self.runner_argument.topic,
            role_name="PureRAG",
            role_description="",
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=self.rm,
            callback_handler=self.callback_handler,
        )
        self.moderator: Moderator = Moderator(
            topic=self.runner_argument.topic,
            role_name="Moderator",
            role_description="",
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            encoder=self.encoder,
            callback_handler=self.callback_handler,
        )
        self.general_knowledge_provider = CoStormExpert(
            topic=self.runner_argument.topic,
            role_name="General Knowledge Provider",
            role_description="Focus on broadly covering the basic facts about the question.",
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=self.rm,
            callback_handler=self.callback_handler,
        )
        self.generate_expert_module = GenerateExpertModule(
            engine=self.lm_config.discourse_manage_lm
        )
        self.next_turn_moderator_override = False

    def serialize_experts(self) -> List[Dict]:
        return [
            {
                "topic": expert.topic,
                "role_name": expert.role_name,
                "role_description": expert.role_description,
            }
            for expert in self.experts
        ]

    def deserialize_experts(self, data: List[Dict]):
        for expert_data in data:
            self.experts.append(
                CoStormExpert(
                    topic=expert_data["topic"],
                    role_name=expert_data["role_name"],
                    role_description=expert_data["role_description"],
                    lm_config=self.lm_config,
                    runner_argument=self.runner_argument,
                    logging_wrapper=self.logging_wrapper,
                    rm=self.rm,
                    callback_handler=self.callback_handler,
                )
            )

    def _should_generate_question(
        self, conversation_history: List[ConversationTurn]
    ) -> bool:
        consecutive_non_questioning_turn = 0
        for conv_turn in reversed(conversation_history):
            if conv_turn.utterance_type not in [
                "Original Question",
                "Information Request",
            ]:
                consecutive_non_questioning_turn += 1
            else:
                break
        return (
            consecutive_non_questioning_turn
            >= self.runner_argument.moderator_override_N_consecutive_answering_turn
        )

    def _parse_expert_names_to_agent(self, expert_descriptions: Union[str, List[str]]):
        if type(expert_descriptions) == str:
            expert_descriptions = [expert_descriptions]
        agents: CoStormExpert = []
        for expert_name in expert_descriptions:
            role_name, role_description = expert_name.split(":")
            role_name = role_name.strip()
            role_description = role_description.strip()
            new_costorm_expert = CoStormExpert(
                topic=self.runner_argument.topic,
                role_name=role_name,
                role_description=role_description,
                lm_config=self.lm_config,
                runner_argument=self.runner_argument,
                logging_wrapper=self.logging_wrapper,
                rm=self.rm,
                callback_handler=self.callback_handler,
            )
            agents.append(new_costorm_expert)
        return agents

    def create_researcher(self, role_name: str, role_description: str):
        """
        Create a Researcher agent with the specified role name and description.

        Args:
            role_name (str): The name of the researcher role (e.g., "Innovation Scientist").
            role_description (str): A description of the researcher's approach and focus.

        Returns:
            Researcher: A new Researcher agent configured for the current topic.
        """
        researcher = Researcher(
            topic=self.runner_argument.topic,
            role_name=role_name,
            role_description=role_description,
            lm_config=self.lm_config,
            runner_argument=self.runner_argument,
            logging_wrapper=self.logging_wrapper,
            rm=self.rm,
            callback_handler=self.callback_handler,
        )
        return researcher

    def _update_expert_list_from_utterance(self, focus: str, background_info: str):
        expert_names = self.generate_expert_module(
            topic=self.runner_argument.topic,
            background_info=background_info,
            focus=focus,
            num_experts=self.runner_argument.max_num_round_table_experts,
        ).experts
        self.experts = self._parse_expert_names_to_agent(expert_names)

    def _is_last_turn_questioning(self, conversation_history: List[ConversationTurn]):
        return conversation_history and conversation_history[-1].utterance_type in [
            "Original Question",
            "Information Request",
        ]

    def get_next_turn_policy(
        self,
        conversation_history: List[ConversationTurn],
        dry_run=False,
        simulate_user=False,
        simulate_user_intent: str = None,
    ) -> TurnPolicySpec:
        next_turn_policy = TurnPolicySpec()
        if simulate_user:
            self.simulated_user.intent = simulate_user_intent
            next_turn_policy.agent = self.simulated_user
        elif self.runner_argument.rag_only_baseline_mode:
            assert self.conversation_history[-1].role == "Guest"
            next_turn_policy.agent = self.pure_rag_agent
        elif self.next_turn_moderator_override:
            next_turn_policy.agent = self.moderator
            if not dry_run:
                self.next_turn_moderator_override = False
        elif (
            not self.runner_argument.disable_moderator
            and self._should_generate_question(conversation_history)
        ):
            next_turn_policy.agent = self.moderator
            next_turn_policy.should_reorganize_knowledge_base = True
        # experts RAG gen
        else:
            next_turn_policy.agent = self.general_knowledge_provider
            if (
                not self._is_last_turn_questioning(conversation_history)
                and not self.runner_argument.disable_multi_experts
            ):
                if dry_run:
                    next_turn_policy.agent = self.experts[0]
                else:
                    next_turn_policy.agent = self.experts.pop(0)
                    self.experts.append(next_turn_policy.agent)
            next_turn_policy.should_update_experts_list = (
                self._is_last_turn_questioning(conversation_history)
                and not self.runner_argument.disable_multi_experts
            )
            next_turn_policy.should_polish_utterance = True
        return next_turn_policy

    def generate_potential_answers(self):
        """
        Generate a potential answer from the specified number of experts on the panel.

        Args:
            expert_indices (Optional[List[int]], optional): Each index corresponds to each expert in the panel. Defaults to None,
                                                          which means all experts on the panel will generate answers.

        Returns:
            _type_: _description_
        """
        return self.invoke_experts(agent_indices=list(range(len(self.experts))))

    def add_researcher_to_discussion(
        self,
        role_name: str = "Research Scientist",
        role_description: str = "Specializes in generating novel research ideas and experimental plans",
    ):
        """
        Add a Researcher agent to the current discussion.

        Args:
            role_name (str, optional): The name of the researcher role. Defaults to "Research Scientist".
            role_description (str, optional): Description of the researcher's focus. Defaults to a general description.

        Returns:
            int: The index of the newly added researcher in the experts list.
        """
        researcher = self.create_researcher(role_name, role_description)
        self.experts.append(researcher)
        return len(self.experts) - 1  # Return the index of the newly added researcher

    def query_raw(self, question_text: str):
        """Take a question and answer it with the PureRAG method without discourse management"""
        if self.purerag_agent is None:
            self.purerag_agent = PureRAGAgent(
                topic=self.runner_argument.topic,
                role_name="Assistant",
                role_description="AI assistant that answers user questions",
                lm_config=self.lm_config,
                runner_argument=self.runner_argument,
                logging_wrapper=self.logging_wrapper,
                rm=self.rm,
            )


class CoStormRunner:
    def __init__(
        self,
        args: RunnerArgument,
        lm_config: CollaborativeStormLMConfigs,
        encoder: Encoder,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        self.logging_wrapper = LoggingWrapper(lm_config)
        self.args = args
        self.lm_config = lm_config
        self.encoder = encoder
        self.rm = rm

        # set callback
        self.callback_handler = callback_handler

        # Create retriever if not provided
        if self.rm is None:
            self.rm = BingSearch(
                bing_search_api_key=os.getenv("BING_SEARCH_API_KEY"),
                k=args.retrieve_top_k,
            )

        # init knowledge base with required parameters
        self.knowledge_base = KnowledgeBase(
            topic=args.topic,
            knowledge_base_lm=lm_config.knowledge_base_lm,
            node_expansion_trigger_count=args.node_expansion_trigger_count,
            encoder=encoder,
        )
        self.report = None
        self.conversation_history = []
        self.warmstart_conv_archive = []

        # init discourse manager
        self.discourse_manager = DiscourseManager(
            topic=args.topic,
            runner_argument=args,
            lm_config=lm_config,
            encoder=encoder,
            knowledge_base=self.knowledge_base,
            logging_wrapper=self.logging_wrapper,
            rm=rm,
            callback_handler=callback_handler,
        )

    def add_researcher(
        self,
        role_name: str = "Research Scientist",
        role_description: str = "Specializes in generating novel research ideas and experimental plans",
    ):
        """
        Add a Researcher agent to the current discussion.

        Args:
            role_name (str, optional): The name of the researcher role. Defaults to "Research Scientist".
            role_description (str, optional): Description of the researcher's focus. Defaults to a general description.

        Returns:
            int: The index of the newly added researcher in the experts list.
        """
        return self.discourse_manager.add_researcher_to_discussion(
            role_name, role_description
        )

    def get_researcher(self, create_if_missing=True):
        """
        Get a Researcher agent from the current experts list.

        Args:
            create_if_missing (bool, optional): If True, creates a new Researcher if none exists. Defaults to True.

        Returns:
            Researcher: A Researcher agent instance, or None if no Researcher exists and create_if_missing is False.
        """
        # Look for an existing researcher in the experts list
        for expert in self.discourse_manager.experts:
            if isinstance(expert, Researcher):
                return expert

        # Create a new researcher if requested
        if create_if_missing:
            researcher_index = self.add_researcher()
            return self.discourse_manager.experts[researcher_index]

        return None

    def generate_research_idea(self, context=None):
        """
        Generate a novel research idea based on the current knowledge base and conversation.

        This method implements a two-stage approach to research idea generation:
        1. First, it generates multiple brief research idea candidates (5-10 ideas)
        2. Then, it evaluates each candidate, selects the most promising one, and develops it fully

        The result includes both the list of initial candidates and the fully developed selected idea,
        along with reasoning for the selection.

        Args:
            context (str, optional): Additional context to consider when generating the idea.
                                    If None, the last utterance from the conversation history is used.

        Returns:
            str: A comprehensive research idea output containing:
                - List of idea candidates (brief one-sentence descriptions)
                - The selected idea number and selection rationale
                - A fully developed research idea description
        """
        # Wrap the entire operation in a pipeline stage context
        with self.logging_wrapper.log_pipeline_stage("research idea generation"):
            researcher = self.get_researcher()

            # Get the knowledge base summary
            knowledge_summary = self.knowledge_base.get_knowledge_base_summary()

            # Create a formatted version of recent conversation history
            # Include much more conversation context - up to 10 turns or more
            conversation_context = ""
            if hasattr(self, "conversation_history") and self.conversation_history:
                # Use more of the conversation history (up to 20 turns)
                recent_turns = self.conversation_history[-20:]
                conversation_context = "Recent Conversation History:\n"
                for i, turn in enumerate(recent_turns):
                    # Include more details about each turn
                    role = turn.role
                    utterance = turn.utterance
                    utterance_type = (
                        turn.utterance_type
                        if hasattr(turn, "utterance_type")
                        else "Message"
                    )

                    conversation_context += f"Turn {len(self.conversation_history) - len(recent_turns) + i + 1} - {role} ({utterance_type}):\n{utterance}\n\n"

            # Add report if available
            report_context = ""
            if hasattr(self, "report") and self.report:
                # Include more of the report (up to 5000 chars)
                report_context = (
                    "Current Research Report:\n" + self.report[:5000] + "...\n\n"
                )

            # Use the provided context or the last utterance from conversation history
            immediate_context = context
            if (
                immediate_context is None
                and hasattr(self, "conversation_history")
                and self.conversation_history
            ):
                immediate_context = self.conversation_history[-1].utterance
                immediate_context = f"Most Recent Utterance: {self.conversation_history[-1].role} - {immediate_context}"

            # Combine all context information
            full_context = f"{conversation_context}\n{report_context}\nImmediate Context: {immediate_context or ''}"

            # Generate the idea using the new two-stage approach in the Researcher class
            # This will:
            # 1. Generate multiple brief idea candidates
            # 2. Select and develop the most promising idea
            # 3. Return a formatted output with candidates, selection reasoning, and developed idea
            return researcher.generate_idea(
                topic=self.args.topic,
                knowledge_summary=knowledge_summary,
                conversation_context=full_context,
            )

    def assess_research_idea(self, idea):
        """
        Assess a research idea for novelty, feasibility, value, and relevance.

        Args:
            idea (str): The research idea to assess

        Returns:
            str: A detailed assessment of the idea
        """
        # Wrap in pipeline stage context
        with self.logging_wrapper.log_pipeline_stage("research idea assessment"):
            researcher = self.get_researcher()
            return researcher.assess_idea(idea=idea, topic=self.args.topic)

    def create_experimental_plan(self, idea, assessment=None):
        """
        Create an experimental plan to test a research idea.

        Args:
            idea (str): The research idea to develop a plan for
            assessment (str, optional): An assessment of the idea. If None, one will be generated.

        Returns:
            str: A detailed experimental plan
        """
        # Wrap in pipeline stage context
        with self.logging_wrapper.log_pipeline_stage("experimental plan creation"):
            researcher = self.get_researcher()

            # Generate an assessment if none is provided
            if assessment is None:
                assessment = self.assess_research_idea(idea)

            return researcher.create_experimental_plan(idea=idea, assessment=assessment)

    def refine_research_idea(self, original_idea, feedback, previous_plan=None):
        """
        Refine a research idea based on specific feedback.

        Args:
            original_idea (str): The original research idea to refine
            feedback (str): Feedback containing critiques, suggestions, or questions about the idea
            previous_plan (str, optional): Previous experimental plan if available

        Returns:
            str: A refined version of the research idea that addresses the feedback
        """
        # Wrap in pipeline stage context
        with self.logging_wrapper.log_pipeline_stage("research idea refinement"):
            researcher = self.get_researcher()

            # Create a formatted version of recent conversation history for additional context
            conversation_context = ""
            if hasattr(self, "conversation_history") and self.conversation_history:
                # Limit to last 5 turns to keep context manageable
                recent_turns = self.conversation_history[-5:]
                conversation_context = "Recent Conversation Context:\n"
                for i, turn in enumerate(recent_turns):
                    conversation_context += f"{turn.role}: {turn.utterance}\n\n"

            # Add report if available for additional research context
            report_context = ""
            if hasattr(self, "report") and self.report:
                report_context = (
                    "Current Research Report Context:\n"
                    + self.report[:2000]
                    + "...\n\n"
                )

            # Add previous experimental plan if available
            plan_context = ""
            if previous_plan:
                plan_context = "Previous Experimental Plan:\n" + previous_plan + "\n\n"

            # Format original idea for reference (even though it's passed as a parameter)
            idea_context = "Original Research Idea:\n" + original_idea + "\n\n"

            # Enhance feedback with additional context
            enhanced_feedback = f"{feedback}\n\n{idea_context}{conversation_context}\n{report_context}\n{plan_context}"

            return researcher.refine_idea(
                original_idea=original_idea,
                feedback=enhanced_feedback,
                topic=self.args.topic,
            )

    def research_pipeline(
        self, context=None, add_to_conversation=True, refine_idea_from_assessment=False
    ):
        """
        Run a complete research pipeline: generate multiple idea candidates, select/develop the best one,
        assess it, and create an experimental plan.

        The pipeline now uses a two-stage idea generation approach:
        1. First, generate multiple brief research idea candidates (5-10 ideas)
        2. Then, evaluate each candidate, select the most promising one, and develop it fully

        The pipeline leverages comprehensive conversation history and previous research to provide
        rich context for idea generation and refinement, making the research process more cohesive and informed.

        Args:
            context (str, optional): Additional context to consider when generating the idea.
            add_to_conversation (bool, optional): Whether to add the research output to the conversation history.
                                                 Defaults to True.
            refine_idea_from_assessment (bool, optional): Whether to use the assessment as feedback to refine
                                                         the original idea before creating the plan.
                                                         Defaults to False.

        Returns:
            dict: A dictionary containing:
                - idea: The complete idea output with candidates and developed idea
                - assessment: Assessment of the selected and developed idea
                - plan: Experimental plan based on the idea
                - refined_idea: (If applicable) A refined version of the idea
                - preliminary_plan: (If applicable) The preliminary plan created before refinement
        """
        # Wrap the entire pipeline in a single context
        with self.logging_wrapper.log_pipeline_stage("complete research pipeline"):
            # Generate initial ideas and select the best one
            idea_output = self.generate_research_idea(context=context)

            # Extract the developed idea for assessment
            idea_parts = idea_output.split("## Developed Research Idea:")
            if len(idea_parts) > 1:
                developed_idea = idea_parts[1].strip()
                # Include selection rationale for context
                selection_parts = idea_output.split("## Selection Rationale:")
                if len(selection_parts) > 1:
                    selection_rationale = (
                        selection_parts[1]
                        .split("## Developed Research Idea:")[0]
                        .strip()
                    )
                    idea_for_assessment = f"# Research Idea\n{developed_idea}\n\n# Selection Context\n{selection_rationale}"
                else:
                    idea_for_assessment = f"# Research Idea\n{developed_idea}"
            else:
                # Fallback if parsing fails
                idea_for_assessment = idea_output

            # Assess the idea
            assessment = self.assess_research_idea(idea_for_assessment)

            # Optionally refine the idea based on the assessment
            refined_idea = None
            if refine_idea_from_assessment:
                # Generate a preliminary plan based on the original idea to provide context for refinement
                preliminary_plan = self.create_experimental_plan(
                    idea_for_assessment, assessment
                )

                # Use the assessment as feedback to refine the idea, with the preliminary plan as context
                refined_idea = self.refine_research_idea(
                    original_idea=idea_for_assessment,
                    feedback=f"Assessment of Original Idea:\n{assessment}",
                    previous_plan=preliminary_plan,
                )

                # Create the final plan based on the refined idea
                plan = self.create_experimental_plan(refined_idea, assessment)
            else:
                # Create plan based on the original idea
                plan = self.create_experimental_plan(idea_for_assessment, assessment)

            # Build result dictionary - now with more detailed components
            result = {
                "idea_output": idea_output,  # Complete formatted output with candidates and selection
                "developed_idea": idea_for_assessment,  # The selected and developed idea
                "assessment": assessment,
                "plan": plan,
                # Extract idea candidates if possible (for easier access)
                "idea_candidates": (
                    idea_output.split("# Selected Idea")[0]
                    .replace("# Research Idea Candidates\n", "")
                    .strip()
                    if "# Research Idea Candidates" in idea_output
                    else ""
                ),
            }

            # Add refined idea to result if applicable
            if refined_idea:
                result["refined_idea"] = refined_idea
                # Also include the preliminary plan in the result for reference
                result["preliminary_plan"] = preliminary_plan

            # Optionally add the research output to conversation history
            if add_to_conversation and hasattr(self, "conversation_history"):
                # Fix the import - use absolute import instead of relative import
                from knowledge_storm.dataclass import ConversationTurn

                # Add the output as a conversation turn from the researcher
                researcher = self.get_researcher()
                if researcher:
                    # Format response for conversation
                    if refined_idea:
                        conversation_response = f"""Research Pipeline Results:
                        
# Original Ideas and Selection
{idea_output}

# Assessment
{assessment}

# Refined Idea
{refined_idea}

# Experimental Plan
{plan}
"""
                    else:
                        conversation_response = f"""Research Pipeline Results:
                        
# Ideas and Selection
{idea_output}

# Assessment
{assessment}

# Experimental Plan
{plan}
"""

                    # Create and add the turn
                    turn = ConversationTurn(
                        role=researcher.role_name,
                        raw_utterance=conversation_response,
                        utterance=conversation_response,
                        utterance_type="Research Output",
                    )
                    self.conversation_history.append(turn)

                    # Update knowledge base with the new information
                    self.knowledge_base.update_from_conv_turn(
                        conv_turn=turn,
                        allow_create_new_node=True,
                        insert_under_root=False,
                    )

            return result

    def to_dict(self):
        result = {
            "topic": self.args.topic,
            "runner_argument": self.args.to_dict(),
            "lm_config": self.lm_config.to_dict(),
            "conversation_history": [
                turn.to_dict() for turn in self.conversation_history
            ],
            "warmstart_conv_archive": [
                turn.to_dict() for turn in self.warmstart_conv_archive
            ],
            "experts": self.discourse_manager.serialize_experts(),
            "knowledge_base": self.knowledge_base.to_dict(),
        }

        # Include the report if it exists
        if hasattr(self, "report") and self.report is not None:
            result["report"] = self.report

        return result

    @classmethod
    def from_dict(cls, data, callback_handler: BaseCallbackHandler = None):
        # Fail if no data is provided
        if not data:
            raise ValueError("No data provided to construct CoStormRunner")

        # Create a new LM config from lm_config data
        lm_config = CollaborativeStormLMConfigs.from_dict(data.get("lm_config", {}))

        # Create logging wrapper
        logging_wrapper = LoggingWrapper(lm_config)

        # Determine if we should use header-based auth for the encoder
        client_id = os.getenv("AZURE_OPENAI_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_OPENAI_CLIENT_SECRET") or os.getenv(
            "AZURE_CLIENT_SECRET"
        )
        tenant_id = os.getenv("AZURE_OPENAI_TENANT_ID") or os.getenv("AZURE_TENANT_ID")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")

        # Create the encoder with appropriate authentication
        encoder = None
        if all([client_id, client_secret, tenant_id]):
            # Use header-based authentication for DOW environment
            auth_headers = get_auth_headers()
            if auth_headers:
                encoder = Encoder(
                    api_key=None,  # Set explicitly to None for header auth
                    api_base=api_base,
                    extra_headers=auth_headers,
                )

        # Fall back to standard encoder if header auth wasn't set up
        if encoder is None:
            encoder = Encoder()

        # Create runner args from the data
        runner_args = RunnerArgument.from_dict(data["runner_argument"])

        # Create retriever with API key from environment variable
        retriever = BingSearch(
            bing_search_api_key=os.getenv("BING_SEARCH_API_KEY"),
            k=runner_args.retrieve_top_k,
        )

        # Initialize CoStormRunner
        costorm_runner = cls(
            args=runner_args,
            lm_config=lm_config,
            encoder=encoder,
            rm=retriever,
            callback_handler=callback_handler,
        )

        # Load data from the dictionary
        if "conversation_history" in data:
            costorm_runner.conversation_history = [
                ConversationTurn.from_dict(turn)
                for turn in data["conversation_history"]
            ]

        if "warmstart_conv_archive" in data:
            costorm_runner.warmstart_conv_archive = [
                ConversationTurn.from_dict(turn)
                for turn in data["warmstart_conv_archive"]
            ]

        if "experts" in data:
            costorm_runner.discourse_manager.deserialize_experts(data["experts"])

        if "knowledge_base" in data:
            costorm_runner.knowledge_base = KnowledgeBase.from_dict(
                data=data["knowledge_base"],
                knowledge_base_lm=costorm_runner.lm_config.knowledge_base_lm,
                node_expansion_trigger_count=costorm_runner.args.node_expansion_trigger_count,
                encoder=costorm_runner.encoder,
            )

        # Load report if it exists
        if "report" in data:
            costorm_runner.report = data["report"]

        return costorm_runner

    def warm_start(self):
        """
        Warm start co-storm system to conduct background information search in order to build shared conceptual space with user.
        This stage is a mini-STORM, spawning multiple LLM agent with different perspective and perform multi-round conversation.
        The knowledge base (i.e. mind map) will be initialize using the collected information.

        It will also generate a first draft of report and use it to produce an engaging and concise conversation presented to the
        user to catch up with system's knowledge about the topic.
        """
        with self.logging_wrapper.log_pipeline_stage(pipeline_stage="warm start stage"):
            if not self.args.rag_only_baseline_mode:
                warm_start_module = WarmStartModule(
                    lm_config=self.lm_config,
                    runner_argument=self.args,
                    logging_wrapper=self.logging_wrapper,
                    rm=self.rm,
                    callback_handler=self.callback_handler,
                )

                (
                    warmstart_conv,
                    warmstart_revised_conv,
                    warmstart_experts,
                ) = warm_start_module.initiate_warm_start(
                    topic=self.args.topic,
                    knowledge_base=self.knowledge_base,
                )
                self.discourse_manager.experts = (
                    self.discourse_manager._parse_expert_names_to_agent(
                        warmstart_experts
                    )
                )
                self.discourse_manager.next_turn_moderator_override = True
                self.conversation_history = (
                    warmstart_revised_conv if warmstart_revised_conv else warmstart_conv
                )
                self.warmstart_conv_archive = warmstart_conv
                self.knowledge_base.reorganize()
            else:
                if self.knowledge_base is None:
                    self.knowledge_base = KnowledgeBase(
                        topic=self.args.topic,
                        knowledge_base_lm=self.lm_config.knowledge_base_lm,
                        node_expansion_trigger_count=self.args.node_expansion_trigger_count,
                        encoder=self.encoder,
                    )
                if self.conversation_history is None:
                    self.conversation_history = []
                conv_turn = (
                    self.discourse_manager.pure_rag_agent.generate_topic_background()
                )
                self.conversation_history.append(conv_turn)
                self.knowledge_base.update_from_conv_turn(
                    conv_turn=conv_turn,
                    allow_create_new_node=True,
                    insert_under_root=self.args.rag_only_baseline_mode,
                )

    def generate_report(self) -> str:
        """
        Generate report leveraging organized collected information in the knowledge base (i.e. mind map).
        The article generation follows the paradigm in STORM paper, where it considers mind map nodes as section names, and generate the report section by section.

        Returns:
            str: A string representing the report, with "#" "##" indicating hierarchical sections and [1][2] indicating references.
        """
        with self.logging_wrapper.log_pipeline_stage(
            f"report generation after conv turn: {len(self.conversation_history)}"
        ):
            with self.logging_wrapper.log_event(
                "report generation stage: generate report"
            ):
                self.report = self.knowledge_base.to_report()
                return self.report

    def dump_logging_and_reset(self):
        return self.logging_wrapper.dump_logging_and_reset()

    def step(
        self,
        user_utterance: str = "",
        simulate_user: bool = False,
        simulate_user_intent: str = "",
    ) -> ConversationTurn:
        """
        Yields a single turn in the conversation flow.

        This method take a user input when user choose to inject an utterance or generates the next system utterance based on the current conversation history and defined discourse policies.
        It handles updating the conversation history, managing expert lists, and interacting with the knowledge base.
        Additionally, it logs each stage of the conversation for monitoring and debugging purposes.

        Args:
            user_utterance (str, optional): The input provided by the user. If provided, this utterance is added directly to the conversation history and returns with no further action.
            simulate_user (bool, optional): This is designed for automatic experiments using a LLM agent to simulate user actions. Flag indicating whether to simulate user behavior. When set to `True`, the system will generate user intents based on predefined simulation logic. Defaults to `False`.
            simulate_user_intent (str, optional): This is designed for automatic experiments using a LLM agent to simulate user actions. Specifies the intent to simulate for the user. This is used when `simulate_user` is `True` to guide the simulated user's responses,

        Returns:
            ConversationTurn: An object representing the latest turn in the conversation.

        Workflow:
            1. User Utterance Handling
                - If `user_utterance` is provided, it is appended to the `conversation_history`

            2. System Utterance Generation
                - If no `user_utterance` is provided, the method proceeds to generate the next system utterance.
                - Determines the next turn policy by consulting the `discourse_manager` with the current conversation history.
                - Generates a new utterance using the agent defined in the turn policy, leveraging the `knowledge_base` and `conversation_history`.
                - If the turn policy indicates that the experts list should be updated, it updates the expert list based on the latest utterances.

            4. Knowledge Base Update
                - Inserts the new turn into the `knowledge_base`, optionally allowing the creation of new nodes or inserting under the root based on the `rag_only_baseline_mode` flag.
                - If the turn policy specifies, it reorganizes the `knowledge_base` to maintain optimal structure and relevance.
        """
        last_conv_turn = self.conversation_history[-1]
        cur_turn_name = f"conv turn: {len(self.conversation_history) + 1}"
        with self.logging_wrapper.log_pipeline_stage(
            pipeline_stage=f"{cur_turn_name} stage"
        ):
            conv_turn = None
            if user_utterance:
                self.discourse_manager.next_turn_moderator_override = False
                conv_turn = ConversationTurn(
                    role="Guest",
                    raw_utterance=user_utterance,
                    utterance_type="Original Question",
                )
                self.conversation_history.append(conv_turn)
            else:
                with self.logging_wrapper.log_event(
                    f"{cur_turn_name}: get turn policy"
                ):
                    if self.callback_handler is not None:
                        self.callback_handler.on_turn_policy_planning_start()
                    turn_policy = self.discourse_manager.get_next_turn_policy(
                        conversation_history=self.conversation_history,
                        simulate_user=simulate_user,
                        simulate_user_intent=simulate_user_intent,
                        dry_run=False,
                    )

                with self.logging_wrapper.log_event(
                    f"{cur_turn_name}: generate utterance"
                ):
                    conv_turn = turn_policy.agent.generate_utterance(
                        knowledge_base=self.knowledge_base,
                        conversation_history=self.conversation_history,
                    )

                if turn_policy.should_update_experts_list:
                    with self.logging_wrapper.log_event(
                        f"{cur_turn_name}: update experts list"
                    ):
                        self.discourse_manager._update_expert_list_from_utterance(
                            focus=last_conv_turn.raw_utterance,
                            background_info=conv_turn.raw_utterance,
                        )

                if conv_turn is not None:
                    self.conversation_history.append(conv_turn)
                    with self.logging_wrapper.log_event(
                        f"{cur_turn_name}: insert into knowledge base"
                    ):
                        if self.callback_handler is not None:
                            self.callback_handler.on_mindmap_insert_start()
                        self.knowledge_base.update_from_conv_turn(
                            conv_turn=conv_turn,
                            allow_create_new_node=True,
                            insert_under_root=self.args.rag_only_baseline_mode,
                        )
                        if self.callback_handler is not None:
                            self.callback_handler.on_mindmap_insert_end()
                if turn_policy.should_reorganize_knowledge_base:
                    with self.logging_wrapper.log_event(
                        f"{cur_turn_name}: reorganize knowledge base"
                    ):
                        if self.callback_handler is not None:
                            self.callback_handler.on_mindmap_reorg_start()
                        self.knowledge_base.reorganize()
        return conv_turn


# Global variables to store token and expiration
_cached_token = None
_token_expiry = None
# Azure AD tokens typically last for ~1 hour
_TOKEN_REFRESH_MARGIN = 300  # Refresh 5 minutes before expiry


def get_auth_headers():
    """
    Helper function to get authentication headers based on available credentials.
    Avoids direct dependency on Azure libraries unless necessary.
    Caches the token to avoid repeated authentication requests.

    Returns:
        dict: A dictionary containing authentication headers if available
    """
    global _cached_token, _token_expiry

    auth_headers = {}

    # Try to get Azure API key if available
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    if api_key:
        auth_headers["api-key"] = api_key

    # Try to get Azure Bearer token only if needed
    client_id = os.getenv("AZURE_OPENAI_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_OPENAI_CLIENT_SECRET") or os.getenv(
        "AZURE_CLIENT_SECRET"
    )
    tenant_id = os.getenv("AZURE_OPENAI_TENANT_ID") or os.getenv("AZURE_TENANT_ID")

    if all([client_id, client_secret, tenant_id]):
        try:
            # Only import azure.identity if we need to get a token
            import importlib.util
            from datetime import datetime, timedelta

            if importlib.util.find_spec("azure.identity"):
                from azure.identity import ClientSecretCredential

                current_time = datetime.now()

                # Check if we have a valid token that's not close to expiration
                if (
                    _cached_token is None
                    or _token_expiry is None
                    or current_time >= _token_expiry
                ):
                    # Token is expired or about to expire, get a new one
                    token = (
                        ClientSecretCredential(
                            client_id=client_id,
                            client_secret=client_secret,
                            tenant_id=tenant_id,
                        )
                        .get_token("https://cognitiveservices.azure.com/.default")
                        .token
                    )
                    _cached_token = token

                    # Calculate when this token will expire (typically 1 hour/3600 seconds)
                    # Setting expiry 5 minutes before actual expiry for safety margin
                    _token_expiry = current_time + timedelta(
                        seconds=1800 - _TOKEN_REFRESH_MARGIN
                    )

                    print(f"New Azure auth token obtained, valid until {_token_expiry}")
                else:
                    # Use cached token
                    token = _cached_token

                auth_headers["Authorization"] = f"Bearer {token}"
        except ImportError:
            # If azure.identity isn't available, continue without token
            pass
        except Exception as e:
            # Handle other exceptions
            print(f"Warning: Could not get Azure authentication token: {e}")

    return auth_headers
