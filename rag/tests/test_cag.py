"""Tests for CAG components."""

import pytest
from pathlib import Path
import tempfile
import shutil

from cag.db_interface import DatabaseInterface, Document


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test.db"

    yield db_path

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_database_creation(temp_db):
    """Test database creation and table setup."""
    db = DatabaseInterface(temp_db)
    db.connect()
    db.create_tables()

    # Check database file exists
    assert temp_db.exists()

    db.disconnect()


def test_add_document(temp_db):
    """Test adding a document to database."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        doc_id = db.add_document(
            content="Test content",
            metadata={"source": "test"}
        )

        assert isinstance(doc_id, int)
        assert doc_id > 0


def test_retrieve_document(temp_db):
    """Test retrieving a document from database."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        # Add document
        content = "Test content for retrieval"
        metadata = {"source": "test", "id": 123}
        doc_id = db.add_document(content, metadata)

        # Retrieve document
        doc = db.get_document(doc_id)

        assert doc is not None
        assert doc.content == content
        assert doc.metadata == metadata


def test_add_multiple_documents(temp_db):
    """Test adding multiple documents."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        documents = [
            ("Document 1", {"id": 1}),
            ("Document 2", {"id": 2}),
            ("Document 3", None)
        ]

        doc_ids = db.add_documents(documents)

        assert len(doc_ids) == 3
        assert all(isinstance(id, int) for id in doc_ids)


def test_get_all_documents(temp_db):
    """Test retrieving all documents."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        # Add documents
        db.add_documents([
            ("Doc 1", None),
            ("Doc 2", None),
            ("Doc 3", None)
        ])

        # Retrieve all
        docs = db.get_all_documents()

        assert len(docs) == 3


def test_search_documents(temp_db):
    """Test document search."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        # Add documents
        db.add_documents([
            ("This is about machine learning", None),
            ("This is about deep learning", None),
            ("This is about cooking", None)
        ])

        # Search
        results = db.search_documents("learning", limit=10)

        assert len(results) == 2
        assert all("learning" in doc.content for doc in results)


def test_delete_document(temp_db):
    """Test deleting a document."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        # Add document
        doc_id = db.add_document("Test doc", None)

        # Delete it
        deleted = db.delete_document(doc_id)
        assert deleted is True

        # Verify it's gone
        doc = db.get_document(doc_id)
        assert doc is None


def test_document_count(temp_db):
    """Test getting document count."""
    with DatabaseInterface(temp_db) as db:
        db.create_tables()

        # Initially empty
        assert db.get_document_count() == 0

        # Add documents
        db.add_documents([("Doc 1", None), ("Doc 2", None)])

        # Check count
        assert db.get_document_count() == 2
