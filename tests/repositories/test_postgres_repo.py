# tests/repositories/test_postgres_repo.py
import pytest
import pytest_asyncio  # For async fixtures
import os
import json
import datetime
import asyncio  # <--- Added import
from typing import AsyncGenerator, Dict, Any, Optional, List, cast
from pydantic import HttpUrl  # <--- Import HttpUrl
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date as date_type
import unittest
import numpy as np
from psycopg.connection_async import AsyncConnection

# Import the class to be tested using the CORRECT path
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.models.paper import Paper

# Import fixtures that will be provided by conftest.py
from tests.conftest import db_pool, repository

# Set up logger for this test module
logger = logging.getLogger(__name__)

# --- Test Database Configuration REMOVED --- #
# Rely on conftest.py for TEST_DB_URL and skipping

# Mark tests as asyncio
pytestmark = pytest.mark.asyncio

# --- Test Data (Corrected to match updated 'Paper' model and 'papers' table schema) ---
# Removed fields not in the updated Paper model (like area, unless added back)
# Ensure authors and categories are lists, not JSON strings
# Ensure published_date is a date object
# Use string URLs for HttpUrl fields, Pydantic will validate

test_paper_data_1 = {
    "pwc_id": "test-paper-1",
    "arxiv_id_base": "2401.00001",
    "title": "Test Paper One",
    "summary": "Summary one.",
    "pdf_url": "http://example.com/pdf/1",  # String URL
    "published_date": date_type(2024, 1, 1),  # date object
    "authors": ["Author A", "Author B"],  # List of strings
    "area": "Computer Vision",  # Keep if area is in the final model/table
    "primary_category": "cs.CV",
    "categories": ["cs.CV", "cs.AI"],  # List of strings
    "pwc_title": "Test Paper One PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-1",  # String URL
    "doi": "10.xxxx/xxxx",
}

test_paper_data_2 = {
    "pwc_id": "test-paper-2",
    "arxiv_id_base": "2402.00002",
    "title": "Test Paper Two",
    "summary": None,
    "pdf_url": None,  # None is valid for Optional[HttpUrl]
    "published_date": date_type(2024, 2, 15),  # date object
    "authors": ["Author D"],  # List of strings
    "area": "NLP",
    "primary_category": "cs.CL",
    "categories": ["cs.CL"],  # List of strings
    "pwc_title": "Test Paper Two PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-2",  # String URL
    "doi": None,
}

test_paper_data_3 = {
    "pwc_id": "test-paper-3",
    "arxiv_id_base": "2312.00003",
    "title": "Test Paper Three (CV)",
    "summary": "Summary three.",
    "pdf_url": "http://example.com/pdf/3",  # String URL
    "published_date": date_type(2023, 12, 25),  # date object
    "authors": ["Author A", "Author E"],  # List of strings
    "area": "Computer Vision",
    "primary_category": "cs.CV",
    "categories": ["cs.CV", "cs.LG"],  # List of strings
    "pwc_title": "Test Paper Three PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-3",  # String URL
    "doi": None,
}

test_paper_data_4 = {
    "pwc_id": "test-paper-4",
    "arxiv_id_base": "2205.00004",
    "title": "Old NLP Paper",
    "summary": "An older NLP paper.",
    "pdf_url": None,
    "published_date": date_type(2022, 5, 10),  # date object
    "authors": ["Author F"],  # List of strings
    "area": "NLP",
    "primary_category": "cs.CL",
    "categories": ["cs.CL", "cs.IR"],  # List of strings
    "pwc_title": "Old NLP Paper PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-4",  # String URL
    "doi": None,
}

# --- Test Cases ---


async def test_get_papers_details_by_ids(repository: PostgresRepository) -> None:
    """Test retrieving details for multiple papers by their IDs."""
    # Insert test data using the repository within the transaction
    # Explicitly create Paper objects to satisfy type checker
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )

    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    assert id1 is not None
    assert id2 is not None
    assert id3 is not None

    # Test fetching existing IDs
    ids_to_fetch = [id2, id1]
    results = await repository.get_papers_details_by_ids(ids_to_fetch)

    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert isinstance(results[1], dict)

    results_map = {r["paper_id"]: r for r in results}
    assert id1 in results_map
    assert id2 in results_map

    # Verify content
    assert results_map[id1]["pwc_id"] == test_paper_data_1["pwc_id"]
    assert results_map[id1]["title"] == test_paper_data_1["title"]
    assert results_map[id1]["authors"] == test_paper_data_1["authors"]

    assert results_map[id2]["pwc_id"] == test_paper_data_2["pwc_id"]
    assert results_map[id2]["summary"] == test_paper_data_2["summary"]

    # Test fetching with non-existent ID
    results_mixed = await repository.get_papers_details_by_ids([id1, 99999])
    assert len(results_mixed) == 1
    assert results_mixed[0]["paper_id"] == id1

    # Test fetching empty list
    results_empty = await repository.get_papers_details_by_ids([])
    assert results_empty == []


async def test_get_paper_details_by_pwc_id_found(
    repository: PostgresRepository,
) -> None:
    """Test retrieving paper details by pwc_id when the paper exists."""
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    id1 = await repository.upsert_paper(paper1)
    assert id1 is not None

    details = await repository.get_paper_details_by_pwc_id(
        str(test_paper_data_1["pwc_id"])
    )

    assert details is not None
    assert isinstance(details, dict)
    assert details["paper_id"] == id1
    assert details["pwc_id"] == test_paper_data_1["pwc_id"]
    assert details["title"] == test_paper_data_1["title"]
    assert details["summary"] == test_paper_data_1["summary"]
    assert details["authors"] == test_paper_data_1["authors"]
    assert details["published_date"] == test_paper_data_1["published_date"]


async def test_get_paper_details_by_pwc_id_not_found(
    repository: PostgresRepository,
) -> None:
    """Test retrieving paper details by pwc_id when the paper does not exist."""
    details = await repository.get_paper_details_by_pwc_id("non-existent-pwc-id")
    assert details is None


async def test_search_papers_by_keyword_no_filters(
    repository: PostgresRepository,
) -> None:
    """Test basic keyword search without filters or specific sorting."""
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )
    paper4 = Paper(
        pwc_id=cast(str, test_paper_data_4["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_4["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_4["title"]),
        summary=cast(Optional[str], test_paper_data_4["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_4["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_4["published_date"]),
        authors=cast(List[str], test_paper_data_4["authors"]),
        area=cast(Optional[str], test_paper_data_4["area"]),
        primary_category=cast(Optional[str], test_paper_data_4["primary_category"]),
        categories=cast(List[str], test_paper_data_4["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_4["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_4["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_4["doi"]),
    )
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert all([id1, id2, id3, id4])  # Check all IDs are valid

    # Search for "Paper"
    results_list, total = await repository.search_papers_by_keyword("Paper")
    assert total == 4
    assert len(results_list) == 4
    # Extract paper_ids and compare sets (order-insensitive)
    returned_ids = {item["paper_id"] for item in results_list}
    expected_ids = {id1, id2, id3, id4}
    assert returned_ids == expected_ids

    # Search for "One"
    results_one_list, total_one = await repository.search_papers_by_keyword("One")
    assert total_one == 1
    assert len(results_one_list) == 1
    assert results_one_list[0]["paper_id"] == id1  # Check the ID in the returned dict

    # Search for non-existent term
    results_none, total_none = await repository.search_papers_by_keyword("NoSuchTerm")
    assert total_none == 0
    assert results_none == []


async def test_search_papers_by_keyword_with_limit_skip(
    repository: PostgresRepository,
) -> None:
    """Test keyword search with limit and skip."""
    # Insert papers 1, 2, 3, 4 (order might be 1, 2, 3, 4 based on insert)
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )
    paper4 = Paper(
        pwc_id=cast(str, test_paper_data_4["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_4["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_4["title"]),
        summary=cast(Optional[str], test_paper_data_4["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_4["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_4["published_date"]),
        authors=cast(List[str], test_paper_data_4["authors"]),
        area=cast(Optional[str], test_paper_data_4["area"]),
        primary_category=cast(Optional[str], test_paper_data_4["primary_category"]),
        categories=cast(List[str], test_paper_data_4["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_4["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_4["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_4["doi"]),
    )
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert id1 is not None
    assert id2 is not None
    assert id3 is not None
    assert id4 is not None

    query = "paper"  # Matches all 4

    # Default sort is published_date DESC, paper_id DESC
    # Expected order of IDs: 2, 1, 3, 4

    # Test fetching the second page (skip 2, limit 2)
    results_list, total_count = await repository.search_papers_by_keyword(
        query=query, skip=2, limit=2, sort_by="published_date", sort_order="desc"
    )

    # Extract paper_ids from the results
    returned_ids = [item["paper_id"] for item in results_list]

    assert total_count == 4
    assert len(results_list) == 2
    # Assert the IDs based on the expected order after pagination
    assert returned_ids == [id3, id4]


async def test_search_papers_by_keyword_with_filters(
    repository: PostgresRepository,
) -> None:
    """Test keyword search with date and area filters."""
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )  # CV, 2024-01-01
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )  # NLP, 2024-02-15
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )  # CV, 2023-12-25
    paper4 = Paper(
        pwc_id=cast(str, test_paper_data_4["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_4["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_4["title"]),
        summary=cast(Optional[str], test_paper_data_4["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_4["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_4["published_date"]),
        authors=cast(List[str], test_paper_data_4["authors"]),
        area=cast(Optional[str], test_paper_data_4["area"]),
        primary_category=cast(Optional[str], test_paper_data_4["primary_category"]),
        categories=cast(List[str], test_paper_data_4["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_4["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_4["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_4["doi"]),
    )  # NLP, 2022-05-10
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert id1 is not None and id2 is not None and id3 is not None and id4 is not None

    # Search for "paper" in NLP area published in 2024
    query = "paper"
    date_from = date_type(2024, 1, 1)
    date_to = date_type(2024, 12, 31)
    area = "NLP"

    results_list, total_count = await repository.search_papers_by_keyword(
        query=query,
        published_after=date_from,
        published_before=date_to,
        filter_area=area,
    )

    # Extract paper_ids from the results
    returned_ids = {item["paper_id"] for item in results_list}

    # Only paper 2 should match all criteria
    assert total_count == 1
    assert len(results_list) == 1
    assert returned_ids == {id2}


async def test_get_all_paper_ids_and_text(repository: PostgresRepository) -> None:
    """Test fetching all paper IDs and summaries/titles."""
    # Insert papers, some with summaries, some without
    # Use explicit keyword args instead of **
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )
    paper4 = Paper(
        pwc_id=cast(str, test_paper_data_4["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_4["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_4["title"]),
        summary=cast(Optional[str], test_paper_data_4["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_4["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_4["published_date"]),
        authors=cast(List[str], test_paper_data_4["authors"]),
        area=cast(Optional[str], test_paper_data_4["area"]),
        primary_category=cast(Optional[str], test_paper_data_4["primary_category"]),
        categories=cast(List[str], test_paper_data_4["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_4["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_4["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_4["doi"]),
    )
    paper_no_summary_title = Paper(  # No summary, use title
        pwc_id="test-paper-5",
        arxiv_id_base="2403.00005",
        title="Paper Without Summary",
        summary=None,
        pdf_url=None,
        published_date=date_type(2024, 3, 1),
        authors=["Author G"],
        area="Other",
        primary_category="cs.XX",
        categories=["cs.XX"],
        pwc_title="Paper Without Summary PWC Title",
        pwc_url=cast(Optional[HttpUrl], "http://paperswithcode.com/paper/test-paper-5"),
        doi=None,
    )
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    id5 = await repository.upsert_paper(paper_no_summary_title)
    assert all([id1, id2, id3, id4, id5])

    # <<< DIAGNOSTIC LOGGING START >>>
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM papers;")
            count_result = await cur.fetchone()
            logger.info(
                f"[DIAGNOSTIC test_get_all_paper_ids_and_text] Count after upsert: {count_result[0] if count_result else 'None'}"
            )
    # <<< DIAGNOSTIC LOGGING END >>>

    results = {}
    logger.info("[test_get_all_paper_ids_and_text] Starting manual iteration...")
    generator = repository.get_all_paper_ids_and_text()
    items_yielded = 0
    try:
        while True:
            paper_id, text_content = await anext(generator)
            logger.info(
                f"[test_get_all_paper_ids_and_text] Yielded: ID={paper_id}, Text='{text_content[:50]}...'"
            )
            results[paper_id] = text_content
            items_yielded += 1
    except StopAsyncIteration:
        logger.info("[test_get_all_paper_ids_and_text] StopAsyncIteration caught.")
        pass  # Expected end of iteration
    except Exception as e:
        logger.error(
            f"[test_get_all_paper_ids_and_text] Error during iteration: {e}",
            exc_info=True,
        )

    logger.info(
        f"[test_get_all_paper_ids_and_text] Manual iteration finished. Items yielded: {items_yielded}"
    )

    # Should return 5 papers
    assert len(results) == 5

    # Check content: Summary if exists, otherwise title
    # Add asserts to ensure IDs are not None before indexing
    assert id1 is not None
    assert results[id1] == test_paper_data_1["summary"]
    assert id2 is not None
    assert results[id2] == test_paper_data_2["title"]  # Fallback to title
    assert id3 is not None
    assert results[id3] == test_paper_data_3["summary"]
    assert id4 is not None
    assert results[id4] == test_paper_data_4["summary"]
    assert id5 is not None
    assert results[id5] == paper_no_summary_title.title  # Fallback to title


# --- Test Fixture for simpler setup --- #
@pytest_asyncio.fixture
async def setup_simple_data(repository: PostgresRepository) -> None:
    """Inserts a couple of papers for tests needing basic data."""
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )
    await repository.upsert_paper(paper1)
    await repository.upsert_paper(paper2)
    logger.info("[setup_simple_data] Finished inserting simple data.")


async def test_fetch_data_cursor(
    repository: PostgresRepository, setup_simple_data: Any
) -> None:
    """Test the async generator fetch functionality."""

    # <<< DIAGNOSTIC LOGGING START >>>
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM papers;")
            count_result = await cur.fetchone()
            logger.info(
                f"[DIAGNOSTIC test_fetch_data_cursor] Count before fetch: {count_result[0] if count_result else 'None'}"
            )
    # <<< DIAGNOSTIC LOGGING END >>>

    query = "SELECT paper_id, pwc_id, title FROM papers ORDER BY paper_id"
    batch_size = 1  # Fetch one row at a time
    count = 0
    ids_seen = set()

    logger.info(
        "[test_fetch_data_cursor] Starting manual iteration 1 (batch_size=1)..."
    )
    generator_b1 = repository.fetch_data_cursor(query, (), batch_size)
    items_yielded_b1 = 0
    try:
        while True:
            row = await anext(generator_b1)
            logger.info(f"[test_fetch_data_cursor] Yielded (b1): {row}")
            assert isinstance(row, dict)
            assert "paper_id" in row
            assert "pwc_id" in row
            assert "title" in row
            assert row["paper_id"] not in ids_seen  # Ensure unique rows yielded
            ids_seen.add(row["paper_id"])
            count += 1
            items_yielded_b1 += 1
            # Test fetching specific paper IDs
            if row["pwc_id"] == test_paper_data_1["pwc_id"]:
                assert row["title"] == test_paper_data_1["title"]
            elif row["pwc_id"] == test_paper_data_2["pwc_id"]:
                assert row["title"] == test_paper_data_2["title"]
    except StopAsyncIteration:
        logger.info("[test_fetch_data_cursor] StopAsyncIteration caught (b1).")
        pass
    except Exception as e:
        logger.error(
            f"[test_fetch_data_cursor] Error during iteration (b1): {e}", exc_info=True
        )

    logger.info(
        f"[test_fetch_data_cursor] Manual iteration 1 finished. Items yielded: {items_yielded_b1}"
    )

    # Should have fetched the two papers inserted by setup_simple_data
    assert count == 2

    # Test with different batch size
    count_b2 = 0
    ids_seen_b2 = set()
    logger.info(
        "[test_fetch_data_cursor] Starting manual iteration 2 (batch_size=5)..."
    )
    generator_b5 = repository.fetch_data_cursor(query, (), batch_size=5)
    items_yielded_b5 = 0
    try:
        while True:
            row_b2 = await anext(generator_b5)
            logger.info(f"[test_fetch_data_cursor] Yielded (b5): {row_b2}")
            assert isinstance(row_b2, dict)
            assert row_b2["paper_id"] not in ids_seen_b2
            ids_seen_b2.add(row_b2["paper_id"])
            count_b2 += 1
            items_yielded_b5 += 1
    except StopAsyncIteration:
        logger.info("[test_fetch_data_cursor] StopAsyncIteration caught (b5).")
        pass
    except Exception as e:
        logger.error(
            f"[test_fetch_data_cursor] Error during iteration (b5): {e}", exc_info=True
        )

    logger.info(
        f"[test_fetch_data_cursor] Manual iteration 2 finished. Items yielded: {items_yielded_b5}"
    )
    assert count_b2 == 2

    # Test with empty result
    query_empty = "SELECT paper_id FROM papers WHERE pwc_id = %s"
    count_empty = 0
    logger.info(
        "[test_fetch_data_cursor] Starting manual iteration 3 (empty result)..."
    )
    generator_empty = repository.fetch_data_cursor(query_empty, ("no-such-id",), 10)
    items_yielded_empty = 0
    try:
        while True:
            _ = await anext(generator_empty)
            logger.info(
                "[test_fetch_data_cursor] Yielded (empty): Unexpected item!"
            )  # Should not happen
            count_empty += 1
            items_yielded_empty += 1
    except StopAsyncIteration:
        logger.info("[test_fetch_data_cursor] StopAsyncIteration caught (empty).")
        pass  # Expected for empty result
    except Exception as e:
        logger.error(
            f"[test_fetch_data_cursor] Error during iteration (empty): {e}",
            exc_info=True,
        )

    logger.info(
        f"[test_fetch_data_cursor] Manual iteration 3 finished. Items yielded: {items_yielded_empty}"
    )
    assert count_empty == 0


async def test_search_papers_by_keyword_sort_by_date(
    repository: PostgresRepository,
) -> None:
    """Test keyword search sorted by published_date."""
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )  # 2024-01-01
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )  # 2024-02-15
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )  # 2023-12-25
    paper4 = Paper(
        pwc_id=cast(str, test_paper_data_4["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_4["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_4["title"]),
        summary=cast(Optional[str], test_paper_data_4["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_4["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_4["published_date"]),
        authors=cast(List[str], test_paper_data_4["authors"]),
        area=cast(Optional[str], test_paper_data_4["area"]),
        primary_category=cast(Optional[str], test_paper_data_4["primary_category"]),
        categories=cast(List[str], test_paper_data_4["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_4["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_4["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_4["doi"]),
    )  # 2022-05-10
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert id1 is not None and id2 is not None and id3 is not None and id4 is not None

    query = "paper"  # Matches all

    # Sort by date descending (default)
    results_list_desc, total_count_desc = await repository.search_papers_by_keyword(
        query=query, sort_by="published_date", sort_order="desc"
    )
    returned_ids_desc = [item["paper_id"] for item in results_list_desc]
    assert total_count_desc == 4
    assert returned_ids_desc == [id2, id1, id3, id4]

    # Sort by date ascending
    results_list_asc, total_count_asc = await repository.search_papers_by_keyword(
        query=query, sort_by="published_date", sort_order="asc"
    )
    returned_ids_asc = [item["paper_id"] for item in results_list_asc]
    assert total_count_asc == 4
    assert returned_ids_asc == [id4, id3, id1, id2]


async def test_get_paper_details_by_id_integration(
    repository: PostgresRepository,
) -> None:
    """Test getting details by the internal paper_id."""
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    inserted_id = await repository.upsert_paper(paper1)
    assert inserted_id is not None

    # Fetch by the returned ID
    details = await repository.get_paper_details_by_id(inserted_id)
    assert details is not None
    assert isinstance(details, dict)
    assert details["paper_id"] == inserted_id
    assert details["pwc_id"] == test_paper_data_1["pwc_id"]
    assert details["title"] == test_paper_data_1["title"]
    assert details["authors"] == test_paper_data_1["authors"]

    # Test fetching non-existent internal ID
    details_none = await repository.get_paper_details_by_id(999999)
    assert details_none is None


# Add test for upsert logic (update existing)
async def test_upsert_paper_update(repository: PostgresRepository) -> None:
    """Test that upsert updates an existing paper based on pwc_id."""
    # Insert initial version
    paper_initial = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    id_initial = await repository.upsert_paper(paper_initial)
    assert id_initial is not None

    # Create updated data with the same pwc_id
    updated_data = test_paper_data_1.copy()
    updated_data["title"] = "Updated Test Paper One Title"
    updated_data["summary"] = "Updated summary."
    updated_data["authors"] = ["Author A", "Author B", "Author C"]  # Example update
    paper_updated = Paper(
        pwc_id=cast(str, updated_data["pwc_id"]),
        arxiv_id_base=cast(Optional[str], updated_data["arxiv_id_base"]),
        title=cast(Optional[str], updated_data["title"]),
        summary=cast(Optional[str], updated_data["summary"]),
        pdf_url=cast(Optional[HttpUrl], updated_data["pdf_url"]),
        published_date=cast(Optional[date_type], updated_data["published_date"]),
        authors=cast(List[str], updated_data["authors"]),
        area=cast(Optional[str], updated_data["area"]),
        primary_category=cast(Optional[str], updated_data["primary_category"]),
        categories=cast(List[str], updated_data["categories"]),
        pwc_title=cast(Optional[str], updated_data["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], updated_data["pwc_url"]),
        doi=cast(Optional[str], updated_data["doi"]),
    )

    # Perform upsert again
    id_updated = await repository.upsert_paper(paper_updated)
    assert id_updated == id_initial  # Should return the same paper_id

    # Fetch details and verify update
    details = await repository.get_paper_details_by_id(id_initial)
    assert details is not None
    assert details["title"] == "Updated Test Paper One Title"
    assert details["summary"] == "Updated summary."
    assert (
        details["pwc_id"] == test_paper_data_1["pwc_id"]
    )  # pwc_id should remain the same


async def test_get_all_papers_for_sync_empty_db(repository: PostgresRepository) -> None:
    """测试在数据库为空时获取所有论文。"""
    # 先清空数据库
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
    
    # 测试方法
    results = await repository.get_all_papers_for_sync()
    
    # 验证结果
    assert isinstance(results, list)
    assert len(results) == 0


async def test_get_all_papers_for_sync_with_data(repository: PostgresRepository) -> None:
    """测试获取所有有摘要的论文。"""
    # 创建测试数据
    paper1 = Paper(
        pwc_id="test-sync-1",
        title="Paper With Summary",
        summary="This is a summary for syncing.",
        authors=["Author X"]
    )
    paper2 = Paper(
        pwc_id="test-sync-2",
        title="Paper Without Summary",
        summary="", # 空摘要
        authors=["Author Y"]
    )
    paper3 = Paper(
        pwc_id="test-sync-3",
        title="Paper With Summary 2",
        summary="Another summary for testing sync.",
        authors=["Author Z"]
    )
    
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    
    assert id1 is not None
    assert id2 is not None
    assert id3 is not None
    
    # 测试方法
    results = await repository.get_all_papers_for_sync()
    
    # 验证结果
    assert isinstance(results, list)
    assert len(results) == 2  # 应该只返回有摘要的论文
    
    paper_ids = {result["paper_id"] for result in results}
    assert id1 in paper_ids
    assert id3 in paper_ids
    assert id2 not in paper_ids


async def test_count_papers_empty_db(repository: PostgresRepository) -> None:
    """测试在数据库为空时计数论文。"""
    # 先清空数据库
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
    
    # 测试方法
    count = await repository.count_papers()
    
    # 验证结果
    assert count == 0


async def test_count_papers_with_data(repository: PostgresRepository) -> None:
    """测试计数有数据的论文表。"""
    # 清空数据库
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
    
    # 创建测试数据
    paper1 = Paper(
        pwc_id="test-count-1",
        title="Test Count Paper 1",
        authors=["Author X"]
    )
    paper2 = Paper(
        pwc_id="test-count-2",
        title="Test Count Paper 2",
        authors=["Author Y"]
    )
    
    await repository.upsert_paper(paper1)
    await repository.upsert_paper(paper2)
    
    # 测试方法
    count = await repository.count_papers()
    
    # 验证结果
    assert count == 2


async def test_search_papers_by_keyword_empty_result(repository: PostgresRepository) -> None:
    """测试搜索不存在的关键词。"""
    # 确保有一些测试数据
    paper = Paper(
        pwc_id="test-search-empty",
        title="Specific Title For Test",
        summary="Specific summary for empty search test",
        authors=["Author Empty"]
    )
    await repository.upsert_paper(paper)
    
    # 搜索不存在的关键词
    results, count = await repository.search_papers_by_keyword("NonExistentKeywordXYZ123")
    
    # 验证结果
    assert isinstance(results, list)
    assert len(results) == 0
    assert count == 0


async def test_search_papers_by_keyword_invalid_sort(repository: PostgresRepository) -> None:
    """测试使用无效的排序字段。"""
    # 创建测试数据
    paper = Paper(
        pwc_id="test-sort-invalid",
        title="Test Invalid Sort",
        summary="Test paper for invalid sort test",
        authors=["Author Sort"]
    )
    await repository.upsert_paper(paper)
    
    # 使用无效的排序字段
    results, count = await repository.search_papers_by_keyword(
        "Test", sort_by="invalid_column"  # type: ignore
    )
    
    # 验证结果
    assert isinstance(results, list)
    assert count > 0
    # 应该仍然返回结果，但使用默认排序


async def test_get_hf_models_by_ids_empty_ids(repository: PostgresRepository) -> None:
    """测试使用空ID列表获取HF模型。"""
    results = await repository.get_hf_models_by_ids([])
    
    assert isinstance(results, list)
    assert len(results) == 0


async def test_get_paper_details_by_id_nonexistent(repository: PostgresRepository) -> None:
    """测试获取不存在的论文ID。"""
    # 使用一个不太可能存在的ID
    result = await repository.get_paper_details_by_id(999999999)
    
    assert result is None


async def test_save_paper_success(repository: PostgresRepository) -> None:
    """Test saving a new paper successfully."""
    paper = Paper(
        pwc_id="save-success-1",
        title="Save Success Test",
        arxiv_id_versioned="2202.00001v1",
        arxiv_id_base="2202.00001",
        summary="Summary here",
        pdf_url=HttpUrl("https://arxiv.org/pdf/2202.00001.pdf"),
        authors=["Tester"],
        published_date=date_type(2022, 2, 1),
        categories=["cs.LG"],
    )
    result_id = await repository.upsert_paper(paper)
    assert isinstance(result_id, int)

    # Verify data was inserted
    details = await repository.get_paper_details_by_id(result_id)
    assert details is not None
    assert details["pwc_id"] == "save-success-1"
    assert details["title"] == "Save Success Test"


async def test_count_hf_models(repository: PostgresRepository) -> None:
    """Test counting Hugging Face models (assuming table exists)."""
    # Assuming table hf_models exists and is empty initially
    count_initial = await repository.count_hf_models()
    assert count_initial == 0

    # Add some dummy data (replace with actual upsert if available)
    # This requires knowing the hf_models table structure
    # If direct insertion is complex, skip this part or use a mock
    # For now, just assert initial count is 0 if no data added


async def test_fetch_one_success(repository: PostgresRepository) -> None:
    """Test fetching a single record successfully."""
    # Insert data first
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper_id = await repository.upsert_paper(paper1)
    assert paper_id is not None

    query = "SELECT title, pwc_id FROM papers WHERE paper_id = %s"
    result = await repository.fetch_one(query, (paper_id,))

    assert result is not None
    assert result["pwc_id"] == test_paper_data_1["pwc_id"]
    assert result["title"] == test_paper_data_1["title"]


async def test_fetch_one_not_found(repository: PostgresRepository) -> None:
    """Test fetching a single record that does not exist."""
    query = "SELECT title FROM papers WHERE paper_id = %s"
    result = await repository.fetch_one(query, (99999,))
    assert result is None


async def test_get_tasks_for_papers_empty_ids(repository: PostgresRepository) -> None:
    """Test fetching tasks for papers with empty IDs."""
    result = await repository.get_tasks_for_papers([])
    
    assert isinstance(result, dict)
    assert len(result) == 0


async def test_get_datasets_for_papers_empty_ids(repository: PostgresRepository) -> None:
    """Test fetching datasets for papers with empty IDs."""
    result = await repository.get_datasets_for_papers([])
    
    assert isinstance(result, dict)
    assert len(result) == 0


async def test_get_repositories_for_papers_empty_ids(repository: PostgresRepository) -> None:
    """Test fetching repositories for papers with empty IDs."""
    result = await repository.get_repositories_for_papers([])
    
    assert isinstance(result, dict)
    assert len(result) == 0


async def test_close_connection_pool(repository: PostgresRepository) -> None:
    """Test closing the connection pool."""
    # Save original pool to restore
    original_pool = repository.pool
    
    # Create mock pool
    mock_pool = AsyncMock()
    repository.pool = mock_pool
    
    try:
        # Test close method
        await repository.close()
        
        # Verify close method was called
        mock_pool.close.assert_called_once()
    finally:
        # Restore original pool
        repository.pool = original_pool


async def test_get_tasks_for_papers_with_tasks(repository: PostgresRepository) -> None:
    """Test fetching tasks for papers with tasks."""
    # First empty the database
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
            await cur.execute("TRUNCATE pwc_tasks RESTART IDENTITY CASCADE")
    
    # Create test data
    paper1 = Paper(
        pwc_id="test-tasks-1",
        title="Test Tasks Paper 1",
        authors=["Author Tasks"]
    )
    paper2 = Paper(
        pwc_id="test-tasks-2",
        title="Test Tasks Paper 2",
        authors=["Author Tasks"]
    )
    
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    assert id1 is not None
    assert id2 is not None
    
    # Add tasks
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO pwc_tasks (paper_id, task_name) VALUES (%s, %s), (%s, %s), (%s, %s)",
                (id1, "Task A", id1, "Task B", id2, "Task C")
            )
    
    # Test method
    result = await repository.get_tasks_for_papers([id1, id2])
    
    # Verify result
    assert isinstance(result, dict)
    assert len(result) == 2
    assert set(result[id1]) == {"Task A", "Task B"}
    assert set(result[id2]) == {"Task C"}

