"""
Main SimGRAG engine.

Integrates pattern generation, subgraph retrieval, and answer generation
to provide end-to-end Knowledge Graph-driven RAG.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import ollama
from loguru import logger

from .pattern_generator import PatternGenerator, PatternGraph
from .subgraph_retriever import SubgraphRetriever, Subgraph, KnowledgeGraph
from .gsd_calculator import GSDCalculator
from common.config import SimGRAGConfig
from common.utils import setup_logging


class SimGRAGEngine:
    """
    Main engine for SimGRAG question answering.

    Implements the complete SimGRAG pipeline:
    1. Query-to-Pattern Alignment
    2. Pattern-to-Subgraph Alignment
    3. Verbalized Subgraph-Augmented Generation
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        config: Optional[SimGRAGConfig] = None,
        gsd_calculator: Optional[GSDCalculator] = None
    ):
        """
        Initialize SimGRAG engine.

        Args:
            knowledge_graph: Knowledge graph for retrieval
            config: Configuration (uses defaults if None)
            gsd_calculator: Optional pre-configured GSD calculator
        """
        self.config = config or SimGRAGConfig()
        self.kg = knowledge_graph

        # Initialize components
        logger.info("Initializing SimGRAG components...")

        self.pattern_generator = PatternGenerator(
            llm_model=self.config.llm_model,
            max_triples=self.config.max_pattern_size,
            use_few_shot=True
        )

        if gsd_calculator is None:
            self.gsd_calculator = GSDCalculator(
                model_name=self.config.embedding_model,
                device="cuda"
            )
        else:
            self.gsd_calculator = gsd_calculator

        self.subgraph_retriever = SubgraphRetriever(
            knowledge_graph=self.kg,
            gsd_calculator=self.gsd_calculator,
            top_k=self.config.top_k
        )

        logger.info("SimGRAG engine initialized successfully")

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using SimGRAG.

        Args:
            query: User question
            top_k: Number of subgraphs to retrieve (uses config if None)
            verbose: Whether to log detailed progress

        Returns:
            Dictionary with:
                - answer: Generated answer
                - pattern: Pattern graph
                - subgraphs: Retrieved subgraphs
                - metadata: Additional information

        Example:
            >>> engine = SimGRAGEngine(kg)
            >>> result = engine.answer_query("Who directed The Matrix?")
            >>> print(result['answer'])
        """
        if verbose:
            logger.info(f"Processing query: {query}")

        if top_k is None:
            top_k = self.config.top_k

        # Step 1: Query-to-Pattern Alignment
        if verbose:
            logger.info("Step 1: Generating pattern graph...")

        pattern = self.pattern_generator.generate(query)

        if not pattern.triples:
            logger.warning("No pattern generated, returning empty response")
            return {
                "answer": "I couldn't understand the query structure.",
                "pattern": pattern,
                "subgraphs": [],
                "metadata": {"error": "No pattern generated"}
            }

        if verbose:
            logger.info(f"Pattern generated with {len(pattern)} triples")
            logger.debug(f"Pattern: {pattern}")

        # Step 2: Pattern-to-Subgraph Alignment
        if verbose:
            logger.info("Step 2: Retrieving subgraphs...")

        subgraphs = self.subgraph_retriever.retrieve_top_k(pattern)

        if not subgraphs:
            logger.warning("No subgraphs retrieved")
            return {
                "answer": "I couldn't find relevant information in the knowledge graph.",
                "pattern": pattern,
                "subgraphs": [],
                "metadata": {"error": "No subgraphs retrieved"}
            }

        if verbose:
            logger.info(f"Retrieved {len(subgraphs)} subgraphs")
            logger.info(f"Best subgraph GSD: {subgraphs[0].gsd:.4f}")

        # Step 3: Verbalized Subgraph-Augmented Generation
        if verbose:
            logger.info("Step 3: Generating answer...")

        answer = self._generate_answer(
            query=query,
            subgraphs=subgraphs[:top_k]
        )

        if verbose:
            logger.info(f"Generated answer: {answer[:100]}...")

        return {
            "answer": answer,
            "pattern": pattern,
            "subgraphs": subgraphs[:top_k],
            "metadata": {
                "num_pattern_triples": len(pattern),
                "num_subgraphs": len(subgraphs),
                "best_gsd": subgraphs[0].gsd if subgraphs else None
            }
        }

    def _verbalize_subgraphs(self, subgraphs: List[Subgraph]) -> str:
        """
        Convert subgraphs to natural language.

        Args:
            subgraphs: List of subgraphs

        Returns:
            Verbalized text representation
        """
        verbalized_parts = []

        for i, subgraph in enumerate(subgraphs, 1):
            part = f"Subgraph {i} (relevance: {1.0 - subgraph.gsd:.2f}):\n"

            for triple in subgraph.triples:
                # Convert triple to natural language
                # Format: "head relation tail"
                statement = f"  - {triple.head} {triple.relation.replace('_', ' ')} {triple.tail}\n"
                part += statement

            verbalized_parts.append(part)

        return "\n".join(verbalized_parts)

    def _generate_answer(
        self,
        query: str,
        subgraphs: List[Subgraph]
    ) -> str:
        """
        Generate answer using LLM with verbalized subgraphs.

        Args:
            query: User query
            subgraphs: Retrieved subgraphs

        Returns:
            Generated answer
        """
        # Verbalize subgraphs
        context = self._verbalize_subgraphs(subgraphs)

        # Create prompt
        prompt = f"""Based on the following knowledge graph information, answer the question.

Knowledge Graph Information:
{context}

Question: {query}

Answer the question concisely based on the provided information. If the information is insufficient, say so.

Answer:"""

        try:
            # Generate answer using Ollama
            response = ollama.generate(
                model=self.config.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 256,
                }
            )

            answer = response["response"].strip()
            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"

    def batch_answer(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple queries.

        Args:
            queries: List of queries
            top_k: Number of subgraphs per query

        Returns:
            List of result dictionaries

        Example:
            >>> engine = SimGRAGEngine(kg)
            >>> queries = ["Who directed The Matrix?", "Who starred in Inception?"]
            >>> results = engine.batch_answer(queries)
        """
        logger.info(f"Batch answering {len(queries)} queries")

        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.answer_query(query, top_k=top_k, verbose=False)
            results.append(result)

        return results

    def add_knowledge(
        self,
        triples: List[Any],
        verbose: bool = True
    ) -> None:
        """
        Add knowledge to the knowledge graph.

        Args:
            triples: List of triples (can be Triple objects or tuples)
            verbose: Whether to log progress
        """
        from .pattern_generator import Triple

        # Convert tuples to Triple objects if needed
        triple_objects = []
        for t in triples:
            if isinstance(t, Triple):
                triple_objects.append(t)
            elif isinstance(t, (tuple, list)) and len(t) == 3:
                triple_objects.append(Triple(head=str(t[0]), relation=str(t[1]), tail=str(t[2])))
            else:
                logger.warning(f"Skipping invalid triple: {t}")

        self.kg.add_triples(triple_objects)

        if verbose:
            logger.info(f"Added {len(triple_objects)} triples to knowledge graph")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "knowledge_graph": {
                "num_triples": len(self.kg),
                "num_entities": len(self.kg.graph.nodes()),
                "num_relations": len(set(t.relation for t in self.kg.triples))
            },
            "config": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "top_k": self.config.top_k,
                "max_pattern_size": self.config.max_pattern_size
            },
            "cache": {
                "embedding_cache_size": self.gsd_calculator.get_cache_size()
            }
        }

        return stats
