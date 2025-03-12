import logging
from typing import Callable, List, Union, Optional, Dict, Any

import dspy
import os

from .rm import BingSearch, SemanticScholarRM


class CombinedRetriever(dspy.Retrieve):
    """Retriever that combines results from multiple retrievers.
    
    This retriever forwards queries to multiple underlying retrievers and combines
    their results. It respects the total number of results (k) by distributing them
    across the retrievers.
    
    This class is designed to work seamlessly with the CoStormRunner class and can be
    passed directly as the 'rm' parameter during CoStormRunner initialization.
    
    Example:
        ```python
        # Create the combined retriever
        combined_rm = CombinedRetriever(
            retrievers=[
                BingSearch(k=5, bing_search_api_key=os.getenv("BING_SEARCH_API_KEY")),
                SemanticScholarRM(k=5, semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
            ],
            k=10
        )
        
        # Initialize CoStormRunner with the combined retriever
        runner = CoStormRunner(
            lm_config=lm_config,
            runner_argument=runner_args,
            logging_wrapper=logging_wrapper,
            rm=combined_rm,  # Pass the combined retriever here
            encoder=encoder
        )
        ```
    """

    def __init__(
        self,
        retrievers: List[dspy.Retrieve],
        k: int = 6,
        weights: List[float] = None,
    ):
        """Initialize the combined retriever.
        
        Parameters:
            retrievers: List of retriever instances to combine
            k: Total number of results to return
            weights: Optional list of weights for each retriever (must sum to 1.0)
                If not provided, equal weight will be given to each retriever
        """
        super().__init__(k=k)
        
        if not retrievers:
            raise ValueError("You must provide at least one retriever")
        
        self.retrievers = retrievers
        self.k = k
        
        # Setup weights for distributing k across retrievers
        if weights:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            if abs(sum(weights) - 1.0) > 0.001:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
        else:
            # Equal distribution by default
            self.weights = [1.0 / len(retrievers)] * len(retrievers)
            
        # Calculate k for each retriever based on weights
        self.k_per_retriever = [max(1, int(self.k * w)) for w in self.weights]
        
        # If the sum of k_per_retriever is less than k due to rounding,
        # distribute the remainder to retrievers in order of weight
        remainder = self.k - sum(self.k_per_retriever)
        if remainder > 0:
            # Sort indices by weight, descending
            indices = sorted(range(len(self.weights)), key=lambda i: self.weights[i], reverse=True)
            for i in range(remainder):
                self.k_per_retriever[indices[i]] += 1
                
        self.usage = 0
    
    def get_usage_and_reset(self) -> Dict[str, Any]:
        """Get usage statistics from all retrievers and reset them.
        
        This method collects usage statistics from each underlying retriever 
        and combines them into a single usage dictionary.
        
        Returns:
            A dictionary containing usage statistics for all retrievers
        """
        usage = {}
        
        # Collect usage from all retrievers
        for i, retriever in enumerate(self.retrievers):
            if hasattr(retriever, "get_usage_and_reset"):
                retriever_usage = retriever.get_usage_and_reset()
                # Add retriever usage to combined usage
                usage.update(retriever_usage)
        
        # Add our own usage
        own_usage = self.usage
        self.usage = 0
        usage["CombinedRetriever"] = own_usage
        
        return usage
    
    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with all retrievers for passages matching query or queries.
        
        This method distributes the query to all underlying retrievers and combines 
        their results, respecting the total k value and the distribution weights.
        
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.
            
        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        self.usage += len(queries)
        
        all_results = []
        
        # Call each retriever with its assigned k value
        for i, retriever in enumerate(self.retrievers):
            try:
                # Set k for this retriever
                retriever.k = self.k_per_retriever[i]
                
                # Get results from this retriever
                results = retriever.forward(query_or_queries, exclude_urls)
                
                # Tag results with the retriever source
                for result in results:
                    if 'source' not in result:
                        result['source'] = retriever.__class__.__name__
                
                all_results.extend(results)
            except Exception as e:
                logging.error(f"Error in retriever {retriever.__class__.__name__}: {e}")
        
        # Limit to k total results if we have more than k
        return all_results[:self.k]


class BingSemanticScholarRetriever(CombinedRetriever):
    """Specialized combined retriever that uses both Bing and Semantic Scholar.
    
    This retriever integrates both web search results from Bing and academic paper 
    information from Semantic Scholar. This provides a comprehensive information
    retrieval solution that covers both general web content and academic literature.
    
    It is specifically designed to be used with CoStormRunner as a drop-in replacement
    for the default BingSearch retriever.
    
    Example:
        ```python
        # Create the combined Bing and Semantic Scholar retriever
        retriever = BingSemanticScholarRetriever(
            k=10,                          # Total results to retrieve
            bing_weight=0.6,               # 60% results from Bing, 40% from Semantic Scholar
            bing_search_api_key=os.getenv("BING_SEARCH_API_KEY"),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        )
        
        # Initialize CoStormRunner with the combined retriever
        runner = CoStormRunner(
            lm_config=lm_config,
            runner_argument=runner_args,
            logging_wrapper=logging_wrapper,
            rm=retriever,  # Pass the combined retriever here
            encoder=encoder
        )
        ```
    """
    
    def __init__(
        self,
        k: int = 6,
        bing_weight: float = 0.5,
        bing_search_api_key=None,
        semantic_scholar_api_key=None,
        is_valid_source: Callable = None,
        year_filter: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """Initialize the Bing + Semantic Scholar combined retriever.
        
        Parameters:
            k: Total number of results to return
            bing_weight: Weight for Bing results (0.0-1.0)
                Semantic Scholar weight will be (1.0 - bing_weight)
            bing_search_api_key: Optional API key for Bing Search
                If None, will try to read from BING_SEARCH_API_KEY environment variable
            semantic_scholar_api_key: Optional API key for Semantic Scholar
                If None, will try to read from SEMANTIC_SCHOLAR_API_KEY environment variable
            is_valid_source: Optional function to filter valid sources
            year_filter: Optional dictionary with 'start_year' and 'end_year' keys
                for filtering Semantic Scholar results by publication year
            **kwargs: Additional parameters to pass to both retrievers
        """
        # Calculate Semantic Scholar weight
        semantic_scholar_weight = 1.0 - bing_weight
        
        # Get API keys from environment if not provided
        if bing_search_api_key is None:
            bing_search_api_key = os.environ.get("BING_SEARCH_API_KEY")
            if not bing_search_api_key:
                logging.warning("No Bing Search API key provided. Bing search might not work.")
                
        if semantic_scholar_api_key is None:
            semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            # Semantic Scholar can work without an API key but with rate limits
        
        # Create Bing Search retriever
        bing_search = BingSearch(
            bing_search_api_key=bing_search_api_key,
            k=max(1, int(k * bing_weight)),
            is_valid_source=is_valid_source,
            **kwargs
        )
        
        # Extract Semantic Scholar specific parameters
        semantic_scholar_kwargs = kwargs.copy()
        semantic_scholar_kwargs.pop('min_char_count', None)  # Not used by SemanticScholarRM
        semantic_scholar_kwargs.pop('snippet_chunk_size', None)  # Not used by SemanticScholarRM
        semantic_scholar_kwargs.pop('webpage_helper_max_threads', None)  # Not used by SemanticScholarRM
        semantic_scholar_kwargs.pop('mkt', None)  # Not used by SemanticScholarRM
        semantic_scholar_kwargs.pop('language', None)  # Not used by SemanticScholarRM
        
        # Add year filter if provided
        if year_filter:
            semantic_scholar_kwargs['year_filter'] = year_filter
        
        # Create Semantic Scholar retriever
        semantic_scholar = SemanticScholarRM(
            semantic_scholar_api_key=semantic_scholar_api_key,
            k=max(1, int(k * semantic_scholar_weight)),
            is_valid_source=is_valid_source,
            **semantic_scholar_kwargs
        )
        
        # Initialize with both retrievers
        super().__init__(
            retrievers=[bing_search, semantic_scholar],
            k=k,
            weights=[bing_weight, semantic_scholar_weight]
        )
