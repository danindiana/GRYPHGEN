"""
Pattern graph generation for SimGRAG.

Generates pattern graphs from user queries using LLMs.
The pattern graph captures the semantic structure of the query.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import ollama


@dataclass
class Triple:
    """Represents a knowledge graph triple (head, relation, tail)."""
    head: str
    relation: str
    tail: str

    def __repr__(self) -> str:
        return f"({self.head}, {self.relation}, {self.tail})"


@dataclass
class PatternGraph:
    """
    Pattern graph generated from a query.

    A pattern graph is a set of triples that capture the semantic
    structure of the user's query.
    """
    triples: List[Triple]
    query: str

    def __len__(self) -> int:
        return len(self.triples)

    def __repr__(self) -> str:
        triples_str = "\n  ".join(str(t) for t in self.triples)
        return f"PatternGraph(\n  Query: {self.query}\n  Triples:\n  {triples_str}\n)"


class PatternGenerator:
    """
    Generates pattern graphs from queries using LLMs.

    This implements the Query-to-Pattern Alignment phase of SimGRAG.
    """

    def __init__(
        self,
        llm_model: str = "llama3",
        max_triples: int = 10,
        use_few_shot: bool = True
    ):
        """
        Initialize pattern generator.

        Args:
            llm_model: LLM model to use (default: llama3)
            max_triples: Maximum number of triples in pattern graph
            use_few_shot: Whether to use few-shot examples
        """
        self.llm_model = llm_model
        self.max_triples = max_triples
        self.use_few_shot = use_few_shot

        logger.info(f"PatternGenerator initialized with model: {llm_model}")

    def _create_prompt(self, query: str) -> str:
        """
        Create prompt for LLM to generate pattern graph.

        Args:
            query: User query

        Returns:
            Formatted prompt
        """
        base_prompt = """Given a user query, extract a pattern graph represented as triples.
Each triple should be in the format: (head_entity, relation, tail_entity)

Only output the triples, one per line, in the format: (head, relation, tail)
Do not include any explanations or additional text.
"""

        if self.use_few_shot:
            few_shot_examples = """
Examples:

Query: "Who directed The Matrix?"
Output:
(The Matrix, directed_by, ?)

Query: "What movies did Tom Hanks star in that won an Oscar?"
Output:
(?, starred_in, Tom Hanks)
(?, won, Oscar)

Query: "Who is the CEO of companies founded by Elon Musk?"
Output:
(?, founded_by, Elon Musk)
(?, has_ceo, ?)
"""
            base_prompt += few_shot_examples

        user_query = f"\nQuery: \"{query}\"\nOutput:"

        return base_prompt + user_query

    def generate(self, query: str) -> PatternGraph:
        """
        Generate a pattern graph from a query.

        Args:
            query: User query

        Returns:
            PatternGraph object

        Example:
            >>> generator = PatternGenerator()
            >>> pattern = generator.generate("Who directed The Matrix?")
            >>> print(pattern.triples)
        """
        logger.debug(f"Generating pattern graph for query: {query}")

        prompt = self._create_prompt(query)

        try:
            # Generate pattern using Ollama
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for more deterministic output
                    "top_p": 0.9,
                    "num_predict": 256,
                }
            )

            pattern_text = response["response"].strip()
            triples = self._parse_triples(pattern_text)

            # Limit to max_triples
            if len(triples) > self.max_triples:
                logger.warning(f"Pattern has {len(triples)} triples, limiting to {self.max_triples}")
                triples = triples[:self.max_triples]

            pattern = PatternGraph(triples=triples, query=query)
            logger.info(f"Generated pattern with {len(triples)} triples")

            return pattern

        except Exception as e:
            logger.error(f"Failed to generate pattern: {e}")
            # Return empty pattern as fallback
            return PatternGraph(triples=[], query=query)

    def _parse_triples(self, text: str) -> List[Triple]:
        """
        Parse triples from LLM output.

        Args:
            text: LLM output text

        Returns:
            List of Triple objects
        """
        triples = []

        for line in text.split('\n'):
            line = line.strip()

            # Skip empty lines and non-triple lines
            if not line or not line.startswith('('):
                continue

            try:
                # Remove parentheses and split by comma
                content = line.strip('()').split(',')

                if len(content) >= 3:
                    head = content[0].strip()
                    relation = content[1].strip()
                    tail = content[2].strip()

                    triple = Triple(head=head, relation=relation, tail=tail)
                    triples.append(triple)

            except Exception as e:
                logger.warning(f"Failed to parse triple from line: {line} - {e}")
                continue

        return triples

    def batch_generate(self, queries: List[str]) -> List[PatternGraph]:
        """
        Generate pattern graphs for multiple queries.

        Args:
            queries: List of user queries

        Returns:
            List of PatternGraph objects

        Example:
            >>> generator = PatternGenerator()
            >>> queries = ["Who directed The Matrix?", "Who starred in Inception?"]
            >>> patterns = generator.batch_generate(queries)
        """
        logger.info(f"Batch generating patterns for {len(queries)} queries")

        patterns = []
        for query in queries:
            pattern = self.generate(query)
            patterns.append(pattern)

        return patterns
