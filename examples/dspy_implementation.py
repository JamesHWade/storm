# TODO: Add outline agent
# TODO: Morph writer to write individual sections
# TODO: Guide article structure more
# TODO: Fix semantic scholar to return more info per result
# TODO: Fix combined retriever to have both sources
# TODO: User DSPy to generate experts based on topic

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Union

import backoff
import dspy
import requests
from dotenv import load_dotenv

import mlflow

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("DSPy")

load_dotenv()

# --- LM Configuration ---
lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
dspy.configure(lm=lm)


# --- DSPy Signatures ---

class ExpertSignature(dspy.Signature):
    """
    Generate an expert discourse turn (question or answer with citations)
    based on the provided topic context, expert persona, and conversation history.
    """
    persona = dspy.InputField(desc="Expert persona", format=str)
    topic_context = dspy.InputField(desc="Current topic and background (including retrieved context)")
    discourse_history = dspy.InputField(desc="Conversation so far")
    utterance = dspy.OutputField(desc="Expert utterance with citations", format=str)


class ModeratorSignature(dspy.Signature):
    """
    Generate a steering question to guide the conversation.
    """
    topic_context = dspy.InputField(desc="Overall topic context")
    discourse_history = dspy.InputField(desc="Conversation so far")
    moderator_question = dspy.OutputField(desc="Steering question", format=str)


class MindMapSignature(dspy.Signature):
    """
    Decide how to update the hierarchical mind map with new information.
    Expected outputs: 'insert', 'step: child_node', or 'create: new_node'.
    """
    info = dspy.InputField(desc="New information (with context)", format=str)
    current_structure = dspy.InputField(desc="Current mind map structure", format=str)
    decision = dspy.OutputField(desc="Decision", format=str)


class ReportSignature(dspy.Signature):
    """
    Generate the final comprehensive report with citations,
    based on the updated mind map and full conversation history.
    """
    mind_map = dspy.InputField(desc="Hierarchical mind map", format=str)
    discourse_history = dspy.InputField(desc="Full conversation history", format=str)
    final_report = dspy.OutputField(desc="Final report with citations", format=str)


class SummarizerSignature(dspy.Signature):
    """
    Summarize the conversation history into a brief summary.
    """
    conversation = dspy.InputField(desc="Full conversation history", format=str)
    summary = dspy.OutputField(desc="Brief summary of the discussion", format=str)

# --- DSPy Modules for each role ---


class ExpertAgent(dspy.Module):
    def __init__(self, persona: str, retriever: Optional[dspy.Retrieve] = None):
        super().__init__()
        self.persona = persona
        self.generator = dspy.ChainOfThought(ExpertSignature)
        self.retriever = retriever

    def forward(self, topic_context: str, discourse_history: str):
        additional_context = ""
        retrieved_results = []
        if self.retriever is not None:
            retrieved_results = self.retriever.forward(topic_context)
            snippets = [r.get("snippets", [""])[0] for r in retrieved_results if r.get("snippets")]
            additional_context = " ".join(snippets)
        full_context = f"{topic_context}\nAdditional context: {additional_context}"
        prediction = self.generator(persona=self.persona,
                                      topic_context=full_context,
                                      discourse_history=discourse_history)
        return prediction.utterance, retrieved_results


class ModeratorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(ModeratorSignature)

    def forward(self, topic_context: str, discourse_history: str) -> str:
        prediction = self.generator(topic_context=topic_context, discourse_history=discourse_history)
        return prediction.moderator_question


class MindMapAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(MindMapSignature)

    def update(self, info: str, current_structure: str) -> tuple:
        prediction = self.generator(info=info, current_structure=current_structure)
        decision = prediction.decision  # e.g., "insert", "step: child", or "create: new_node"
        updated_structure = current_structure + "\n" + decision
        return updated_structure, decision


class ReportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(ReportSignature)

    def forward(self, mind_map: str, discourse_history: str) -> str:
        prediction = self.generator(mind_map=mind_map, discourse_history=discourse_history)
        return prediction.final_report


class SummarizerAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(SummarizerSignature)

    def forward(self, conversation: str) -> str:
        prediction = self.generator(conversation=conversation)
        return prediction.summary


# --- Combined Retriever Implementation ---


class SemanticScholarRM(dspy.Retrieve):
    """Retrieve information from Semantic Scholar, a search engine for academic papers."""
    def __init__(
        self,
        semantic_scholar_api_key: Optional[str] = None,
        k: int = 3,
        is_valid_source: Optional[Callable] = None,
        max_retries: int = 8,
        fields: Optional[List[str]] = None,
        year_filter: Optional[Dict[str, int]] = None,
        timeout: int = 30,
    ):
        super().__init__(k=k)
        try:
            import semanticscholar as ss
        except ImportError as err:
            raise ImportError(
                "SemanticScholarRM requires `pip install semanticscholar`."
            ) from err

        self.default_fields = [
            "title",
            "abstract",
            "venue",
            "year",
            "authors",
            "url",
            "paperId",
            "externalIds",
            "publicationDate",
            "citationCount",
        ]
        self.fields = fields if fields is not None else self.default_fields
        self.year_filter = year_filter
        self.max_retries = max_retries
        self.timeout = timeout
        self.usage = 0

        if semantic_scholar_api_key:
            self.api_key = semantic_scholar_api_key
            self.client = ss.SemanticScholar(api_key=self.api_key, timeout=self.timeout)
        elif os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
            self.api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
            self.client = ss.SemanticScholar(api_key=self.api_key, timeout=self.timeout)
        else:
            self.api_key = None
            self.client = ss.SemanticScholar(timeout=self.timeout)
            logging.warning(
                "No Semantic Scholar API key provided. Using unauthenticated access with lower rate limits."
            )
        self.is_valid_source = is_valid_source if is_valid_source else lambda x: True

    def get_usage_and_reset(self) -> Dict[str, Any]:
        usage = self.usage
        self.usage = 0
        return {"SemanticScholarRM": usage}

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=1000,
        max_tries=8,
    )
    def _search_papers(self, query: str):
        params = {
            "query": query,
            "limit": self.k * 2,
            "fields": self.fields,
        }
        if self.year_filter and isinstance(self.year_filter, dict):
            if "start_year" in self.year_filter and "end_year" in self.year_filter:
                params["year"] = f"{self.year_filter['start_year']}-{self.year_filter['end_year']}"
            elif "year" in self.year_filter:
                params["year"] = self.year_filter["year"]
        logging.info(f"Searching Semantic Scholar with params: {params}")
        try:
            results = self.client.search_paper(**params)
            if results:
                logging.info(f"Got {len(results)} results from Semantic Scholar API")
            else:
                logging.warning(f"No results for query: {query}")
            return results
        except Exception as api_error:
            logging.warning(f"Error with detailed search, trying simplified: {api_error}")
            simplified_params = {"query": query, "limit": self.k}
            results = self.client.search_paper(**simplified_params)
            return results

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if exclude_urls is None:
            exclude_urls = []
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                results = self._search_papers(query)
                if not results:
                    logging.warning(f"No results for query: {query}")
                    continue
                for paper in results:
                    try:
                        title = getattr(paper, "title", "")
                        abstract = getattr(paper, "abstract", "")
                        paper_id = getattr(paper, "paperId", None)
                        url = getattr(paper, "url", None)
                        if not url:
                            external_ids = getattr(paper, "externalIds", None)
                            if external_ids:
                                doi = getattr(external_ids, "DOI", None)
                                if doi:
                                    url = f"https://doi.org/{doi}"
                                elif getattr(external_ids, "ArXiv", None):
                                    arxiv_id = getattr(external_ids, "ArXiv", None)
                                    if arxiv_id:
                                        url = f"https://arxiv.org/abs/{arxiv_id}"
                            if not url and paper_id:
                                url = f"https://www.semanticscholar.org/paper/{paper_id}"
                        if not url:
                            logging.warning(f"No URL found for paper: {title}")
                            continue
                        if not self.is_valid_source(url) or url in exclude_urls:
                            continue
                        authors = getattr(paper, "authors", [])
                        author_names = []
                        if authors:
                            for author in authors:
                                if hasattr(author, "name"):
                                    author_name = getattr(author, "name", "")
                                    if author_name:
                                        author_names.append(author_name)
                        author_string = ", ".join(author_names)
                        year = getattr(paper, "year", None)
                        year_str = f" ({year})" if year else ""
                        venue = getattr(paper, "venue", None)
                        venue_str = f" - {venue}" if venue else ""
                        citations = getattr(paper, "citationCount", None)
                        citations_str = f" [Citations: {citations}]" if citations is not None else ""
                        snippets = []
                        if abstract:
                            snippets.append(abstract)
                        else:
                            snippets.append(f"Title: {title}")
                        pub_info = f"Authors: {author_string}{year_str}{venue_str}{citations_str}"
                        if pub_info.strip() != "Authors:":
                            snippets.append(pub_info)
                        result = {
                            "title": title,
                            "description": abstract or title,
                            "snippets": snippets,
                            "url": url,
                        }
                        collected_results.append(result)
                        if len(collected_results) >= self.k:
                            break
                    except Exception as paper_error:
                        logging.error(f"Error processing paper: {paper_error}")
                        continue
            except Exception as e:
                logging.error(f"Error searching query '{query}': {e}")
        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(
        self,
        bing_search_api_key: Optional[str] = None,
        k: int = 3,
        is_valid_source: Optional[Callable] = None,
    ):
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError("BING_SEARCH_API_KEY is required.")
        self.bing_search_api_key = bing_search_api_key or os.environ["BING_SEARCH_API_KEY"]
        self.k = k
        self.usage = 0
        self.is_valid_source = is_valid_source if is_valid_source else lambda x: True

    def get_usage_and_reset(self) -> Dict[str, Any]:
        usage = self.usage
        self.usage = 0
        return {"BingSearch": usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Dict[str, Any]]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        collected_results = []
        headers = {"Ocp-Apim-Subscription-Key": self.bing_search_api_key}
        for query in queries:
            try:
                response = requests.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers=headers,
                    params={"q": query, "count": self.k},
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("webPages", {}).get("value", [])
                for r in results:
                    url = r.get("link")
                    if url and self.is_valid_source(url) and url not in exclude_urls:
                        collected_results.append({
                            "title": r.get("name", ""),
                            "description": r.get("snippet", ""),
                            "snippets": [r.get("snippet", "")],
                            "url": url,
                        })
            except Exception as e:
                logging.error(f"Error searching query '{query}': {e}")
        return collected_results


# We'll use the provided SemanticScholarRM (omitted here for brevity; assume it is defined as provided)

# Now, define a CombinedRetriever that integrates BingSearch and SemanticScholarRM.
class CombinedRetriever(dspy.Retrieve):
    def __init__(self, retrievers: List[dspy.Retrieve], k: int = 6, weights: Optional[List[float]] = None):
        super().__init__(k=k)
        if not retrievers:
            raise ValueError("At least one retriever is required.")
        self.retrievers = retrievers
        self.k = k
        self.weights = weights if weights else [1.0 / len(retrievers)] * len(retrievers)
        self.k_per_retriever = [max(1, int(self.k * w)) for w in self.weights]
        remainder = self.k - sum(self.k_per_retriever)
        if remainder > 0:
            indices = sorted(range(len(self.weights)), key=lambda i: self.weights[i], reverse=True)
            for i in range(remainder):
                self.k_per_retriever[indices[i]] += 1
        self.usage = 0

    def get_usage_and_reset(self) -> Dict[str, Any]:
        usage = {}
        for retriever in self.retrievers:
            if hasattr(retriever, "get_usage_and_reset"):
                usage.update(retriever.get_usage_and_reset())
        own_usage = self.usage
        self.usage = 0
        usage["CombinedRetriever"] = own_usage
        return usage

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Dict[str, Any]]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            try:
                retriever.k = self.k_per_retriever[i]
                results = retriever.forward(query_or_queries, exclude_urls)
                for r in results:
                    r["source"] = retriever.__class__.__name__
                all_results.extend(results)
            except Exception as e:
                logging.error(f"Error in retriever {retriever.__class__.__name__}: {e}")
        seen_urls = set()
        dedup_results = []
        for res in all_results:
            url = res.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                dedup_results.append(res)
        return dedup_results[:self.k]


class BingSemanticSearchRetriever(CombinedRetriever):
    """
    A combined retriever that uses both BingSearch and SemanticScholarRM.
    """
    def __init__(
        self,
        k: int = 6,
        bing_weight: float = 0.9,
        bing_search_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        is_valid_source: Optional[Callable] = None,
        **kwargs,
    ):
        semantic_weight = 1.0 - bing_weight
        bing_search = BingSearch(
            bing_search_api_key=bing_search_api_key,
            k=3,
            is_valid_source=is_valid_source,
        )
        # Here, we use the provided SemanticScholarRM version.
        semantic_search = SemanticScholarRM(
            semantic_scholar_api_key=semantic_scholar_api_key,
            k=3,
            is_valid_source=is_valid_source,
            **kwargs,
        )
        super().__init__(retrievers=[bing_search, semantic_search], k=k, weights=[bing_weight, semantic_weight])


# --- Collaborative Report Session Class with Extended Citations Artifact ---
class CollaborativeReportSession:
    """
    Encapsulates a collaborative report writing session.
    Artifacts include conversation history, dynamic mind map, final report, and an elegant citations artifact.
    The citations artifact records details such as Title, Source, Query, Trigger, and Knowledge Base Placement.
    
    Methods:
      - step(user_input=None): Execute one conversation turn.
      - generate_report(): Generate the final report.
      - get_summary(): Return a brief summary of the conversation.
      - get_artifacts(): Return all session artifacts.
      - get_citations(): Return the citations artifact.
      - save_artifacts(filepath): Save artifacts to a file.
      - load_artifacts(filepath): Load artifacts from a file.
    """
    def __init__(self, topic: str, expert_personas: List[str], retriever: Optional[dspy.Retrieve] = None):
        self.topic = topic
        self.conversation_history = ""
        self.mind_map = "Root"
        self.turn_count = 0
        self.final_report = None

        self.expert_agents = [ExpertAgent(persona=p, retriever=retriever) for p in expert_personas]
        self.num_experts = len(self.expert_agents)
        self.moderator_agent = ModeratorAgent()
        self.mind_map_agent = MindMapAgent()
        self.report_agent = ReportAgent()
        self.summarizer_agent = SummarizerAgent()

        # Artifacts to store citation details and mind map updates.
        self.citations = []  # List of dicts with citation keys and metadata

    def step(
        self,
        user_input: Optional[str] = None
    ) -> Optional[str]:
        """Execute one conversation turn.
        If user_input is provided, it's added to the conversation history.
        Otherwise, an expert or moderator takes a turn in the conversation.
        Args:
            user_input: Optional user input to add to the conversation.
        Returns:
            The generated response or user input.
        """
        if user_input:
            # If user input is provided, add it to the conversation history with proper formatting
            self.conversation_history += f"\n\n[User]\n{user_input}"
            return user_input  # Return the user input without further processing

        # Select expert using round-robin.
        expert = self.expert_agents[self.turn_count % self.num_experts]
        expert_turn, retrieved_results = expert.forward(
            self.topic,
            self.conversation_history
        )
        
        # Ensure retrieved_results is always a valid list
        if retrieved_results is None:
            retrieved_results = []
            
        # Log retrieved results for debugging
        logging.debug(f"Retrieved {len(retrieved_results)} results for expert turn")

        # Remove trailing text with citation references inside square brackets
        citation_pattern = r'\[Citations?:(.*?)\]'
        clean_expert_turn = re.sub(citation_pattern, '', expert_turn, flags=re.DOTALL)
        # Format expert response with proper structure
        formatted_expert_turn = f"\n\n[Expert: {expert.persona}]\n{clean_expert_turn}"
        # Add formatted citations at the end if any exist
        if retrieved_results:
            citation_text = "\n\nReferences:"
            for i, result in enumerate(retrieved_results):
                if isinstance(result, dict) and "title" in result:
                    authors = ", ".join(result.get("authors", [])[:3])
                    if len(result.get("authors", [])) > 3:
                        authors += " et al."
                    year = result.get("year", "")
                    title = result.get("title", "")
                    source = result.get("source", "")
                    citation_text += f"\n{i + 1}. {authors} ({year}). {title}. {source}."
            formatted_expert_turn += citation_text
            
            # Add citation references in the text
            formatted_expert_turn += f"\n\n(Citations: {', '.join([f'[{j+1}]' for j in range(len(retrieved_results))])})"
        self.conversation_history += formatted_expert_turn

        # Process cited references from the expert's response if no retriever was used
        if not retrieved_results:
            # Extract citations from the expert's response using regex
            citation_list_pattern = r'Citations?:(.*?)(?=\n\n|\Z)'
            citation_blocks = re.findall(citation_list_pattern, expert_turn, re.DOTALL)
            
            if citation_blocks:
                citation_text = citation_blocks[0].strip()
                citation_entries = citation_text.split('\n')
                
                for entry in citation_entries:
                    match = re.match(r'\[(\d+)\]\s+(.*)', entry.strip())
                    if match:
                        citation_num = match.group(1)
                        citation_info = match.group(2).strip()
                        
                        # Extract author, year, title, source if possible
                        author_year_match = re.match(r'([^(]+)\((\d{4})\)\.?\s+"?([^"]+)"?\.?\s*(.*)', citation_info)
                        if author_year_match:
                            authors = [author_year_match.group(1).strip()]
                            year = author_year_match.group(2)
                            title = author_year_match.group(3).strip().strip('".')
                            source = author_year_match.group(4).strip().strip('.')
                            
                            citation_entry = {
                                "Title": title,
                                "Source": source,
                                "Authors": authors,
                                "Year": year,
                                "URL": "",
                                "Query": self.topic,
                                "Trigger": "Expert turn",
                                "CitationKey": f"[{citation_num}]",
                                "KnowledgeBasePlacement": None
                            }
                            self.citations.append(citation_entry)
                            logging.debug(f"Added citation from text: {title}")

        # Citation entry with detailed information
        for result in retrieved_results:
            # Skip empty or invalid results
            if not result or not isinstance(result, dict):
                continue
                
            # Ensure we have at least a title or source - skip entries with neither
            title = result.get("title", "")
            source = result.get("source", "")
            if not (title or source):
                continue
                
            citation_entry = {
                "Title": title,
                "Source": source,
                "Authors": result.get("authors", []),
                "Year": result.get("year", ""),
                "URL": result.get("url", ""),
                "Query": self.topic,
                "Trigger": "Expert turn",
                "CitationKey": f"[{len(self.citations) + 1}]",
                "KnowledgeBasePlacement": None
            }
            self.citations.append(citation_entry)
            logging.debug(f"Added citation: {title} from {source}")

        # Update the mind map and record the knowledge update.
        updated_mind_map, decision = self.mind_map_agent.update(info=expert_turn, current_structure=self.mind_map)
        self.mind_map = updated_mind_map

        # Every 2 turns, let moderator intervene.
        if self.turn_count % 2 == 1:
            mod_turn = self.moderator_agent.forward(self.topic, self.conversation_history)
            
            # Format moderator response with proper structure
            formatted_mod_turn = f"\n\n[Moderator]\n{mod_turn}"
            
            # Add a reference to the moderator citation
            citation_key = f"[{len(self.citations) + 1}]"
            formatted_mod_turn += f"\n\n(Citation: {citation_key})"
            
            self.conversation_history += formatted_mod_turn
            
            # Add moderator citation entry with complete information
            self.citations.append({
                "Title": "Moderator Summary",
                "Source": "ModeratorAgent",
                "Authors": ["System Moderator"],
                "Year": "",
                "URL": "",
                "Query": self.topic,
                "Trigger": "Moderator turn",
                "CitationKey": citation_key,
                "KnowledgeBasePlacement": None
            })
            updated_mind_map, decision = self.mind_map_agent.update(info=mod_turn, current_structure=self.mind_map)
            self.mind_map = updated_mind_map

        self.turn_count += 1

    def generate_report(self) -> str:
        self.final_report = self.report_agent.forward(self.mind_map, self.conversation_history)
        return self.final_report

    def get_summary(self) -> str:
        summary = self.summarizer_agent.forward(self.conversation_history)
        return summary

    def get_artifacts(self) -> Dict[str, Any]:
        artifacts = {
            "topic": self.topic,
            "conversation_history": self.conversation_history,
            "mind_map": self.mind_map,
            "final_report": self.final_report,
            "turn_count": self.turn_count,
            "citations": self.citations
        }
        return artifacts

    def get_citations(self) -> List[Dict[str, Any]]:
        return self.citations

    def save_artifacts(self, filepath: str):
        artifacts = self.get_artifacts()
        try:
            with open(filepath, "w") as f:
                json.dump(artifacts, f, indent=2)
            logging.info(f"Artifacts saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving artifacts: {e}")

    def load_artifacts(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                artifacts = json.load(f)
            self.topic = artifacts.get("topic", self.topic)
            self.conversation_history = artifacts.get("conversation_history", "")
            self.mind_map = artifacts.get("mind_map", "Root")
            self.final_report = artifacts.get("final_report", None)
            self.turn_count = artifacts.get("turn_count", 0)
            self.citations = artifacts.get("citations", [])
            logging.info(f"Artifacts loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading artifacts: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Instantiate a combined retriever (for example, BingSemanticSearchRetriever).
    # For demonstration, we set it to None. Replace with an actual retriever instance.
    combined_retriever = BingSemanticSearchRetriever()  # e.g., BingSemanticSearchRetriever(...)

    expert_personas = [
        "Data Scientist focusing on remote work trends",
        "Organizational Psychologist studying workplace behavior",
        "Business Strategist with industry insights"
    ]

    session = CollaborativeReportSession(
        topic="The impact of remote work on productivity and innovation",
        expert_personas=expert_personas,
        retriever=combined_retriever,
    )

    # Simulate several conversation steps.
    for _ in range(5):
        session.step()

    # Add an additional user turn.
    session.step(user_input="I'm also interested in its environmental impact.")

    # Generate the final report.
    final_report = session.generate_report()
    print("Final Report:\n", final_report)

    # Print summary.
    summary = session.get_summary()
    print("Summary:\n", summary)

    # Print all artifacts.
    artifacts = session.get_artifacts()
    for key, value in artifacts.items():
        print(f"{key}:\n{value}\n")

    # Print citations artifact.
    citations = session.get_citations()
    print("Citations Artifact:\n", citations)

    # Save and then load artifacts (demonstration).
    session.save_artifacts("session_artifacts.json")
    session.load_artifacts("session_artifacts.json")
