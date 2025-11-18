"""
Database interface for CAG.

Handles document storage and retrieval from SQLite database.
"""

import sqlite3
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
import json


@dataclass
class Document:
    """Represents a document in the database."""
    id: int
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }


class DatabaseInterface:
    """
    Interface for document storage in SQLite.

    Provides methods for storing, retrieving, and managing documents
    for Cache-Augmented Generation.
    """

    def __init__(self, db_path: Path):
        """
        Initialize database interface.

        Args:
            db_path: Path to SQLite database file

        Example:
            >>> db = DatabaseInterface(Path("documents.db"))
            >>> db.create_tables()
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

        logger.info(f"DatabaseInterface initialized with path: {db_path}")

    def connect(self) -> None:
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        logger.debug(f"Connected to database: {self.db_path}")

    def disconnect(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.debug("Disconnected from database")

    def create_tables(self) -> None:
        """
        Create necessary database tables.

        Creates:
        - documents: Main document storage
        - embeddings: Document embeddings (optional)
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create embeddings table (optional, for hybrid approaches)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                document_id INTEGER PRIMARY KEY,
                embedding BLOB,
                model_name TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Create index on content for faster search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_content
            ON documents(content)
        """)

        self.conn.commit()
        logger.info("Database tables created successfully")

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> int:
        """
        Add a document to the database.

        Args:
            content: Document content
            metadata: Optional metadata dictionary
            embedding: Optional embedding vector

        Returns:
            Document ID

        Example:
            >>> db = DatabaseInterface(Path("documents.db"))
            >>> db.connect()
            >>> doc_id = db.add_document("Sample content", {"source": "web"})
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        # Convert metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute(
            "INSERT INTO documents (content, metadata) VALUES (?, ?)",
            (content, metadata_json)
        )

        document_id = cursor.lastrowid

        # Add embedding if provided
        if embedding is not None:
            embedding_blob = json.dumps(embedding).encode()
            cursor.execute(
                "INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)",
                (document_id, embedding_blob)
            )

        self.conn.commit()

        logger.debug(f"Added document with ID: {document_id}")

        return document_id

    def add_documents(
        self,
        documents: List[Tuple[str, Optional[Dict[str, Any]]]]
    ) -> List[int]:
        """
        Add multiple documents in batch.

        Args:
            documents: List of (content, metadata) tuples

        Returns:
            List of document IDs

        Example:
            >>> docs = [("Doc 1", {"source": "web"}), ("Doc 2", None)]
            >>> ids = db.add_documents(docs)
        """
        document_ids = []

        for content, metadata in documents:
            doc_id = self.add_document(content, metadata)
            document_ids.append(doc_id)

        logger.info(f"Added {len(document_ids)} documents to database")

        return document_ids

    def get_document(self, document_id: int) -> Optional[Document]:
        """
        Retrieve a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document object or None if not found
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT id, content, metadata FROM documents WHERE id = ?",
            (document_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        # Parse metadata
        metadata = json.loads(row["metadata"]) if row["metadata"] else None

        # Get embedding if exists
        cursor.execute(
            "SELECT embedding FROM embeddings WHERE document_id = ?",
            (document_id,)
        )

        emb_row = cursor.fetchone()
        embedding = None
        if emb_row:
            embedding = json.loads(emb_row["embedding"].decode())

        return Document(
            id=row["id"],
            content=row["content"],
            metadata=metadata,
            embedding=embedding
        )

    def get_all_documents(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Document]:
        """
        Retrieve all documents.

        Args:
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip

        Returns:
            List of Document objects

        Example:
            >>> docs = db.get_all_documents(limit=100)
            >>> print(f"Retrieved {len(docs)} documents")
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        query = "SELECT id, content, metadata FROM documents"

        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"

        cursor.execute(query)

        documents = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            doc = Document(
                id=row["id"],
                content=row["content"],
                metadata=metadata
            )
            documents.append(doc)

        logger.debug(f"Retrieved {len(documents)} documents")

        return documents

    def search_documents(
        self,
        query: str,
        limit: int = 10
    ) -> List[Document]:
        """
        Search documents by content (simple text search).

        For more advanced search, consider using FTS5 or external search.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT id, content, metadata FROM documents WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        )

        documents = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            doc = Document(
                id=row["id"],
                content=row["content"],
                metadata=metadata
            )
            documents.append(doc)

        logger.debug(f"Search returned {len(documents)} documents")

        return documents

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        cursor.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,)
        )

        self.conn.commit()

        deleted = cursor.rowcount > 0
        logger.debug(f"Deleted document {document_id}: {deleted}")

        return deleted

    def get_document_count(self) -> int:
        """
        Get total number of documents.

        Returns:
            Number of documents
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM documents")
        count = cursor.fetchone()["count"]

        return count

    def clear_all_documents(self) -> None:
        """Delete all documents from the database."""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()

        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM embeddings")

        self.conn.commit()

        logger.warning("All documents cleared from database")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
