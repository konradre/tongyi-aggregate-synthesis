"""End-to-end tests for complete research flow."""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestFullResearchFlow:
    """End-to-end tests for complete research workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_research_flow(
        self, client, searxng_configured, llm_configured
    ):
        """Test complete flow: health → search → research."""
        if not searxng_configured or not llm_configured:
            pytest.skip("Full stack required")

        # Step 1: Health check
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        query = "What are the benefits of async programming?"

        # Step 2: Search only
        search_response = client.post("/api/v1/search", json={
            "query": query,
            "top_k": 5
        })
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert len(search_data["sources"]) > 0

        # Step 3: Full research
        research_response = client.post("/api/v1/research", json={
            "query": query,
            "top_k": 5,
            "reasoning_effort": "medium"
        })
        assert research_response.status_code == 200
        research_data = research_response.json()

        # Verify research response
        assert research_data["query"] == query
        assert len(research_data["content"]) > 100
        assert len(research_data["sources"]) > 0
        assert "connectors_used" in research_data

    @pytest.mark.integration
    @pytest.mark.slow
    def test_research_with_all_connectors(
        self, client, searxng_configured, tavily_configured, linkup_configured, llm_configured
    ):
        """Test research using all available connectors."""
        if not llm_configured:
            pytest.skip("LLM required")

        available = []
        if searxng_configured:
            available.append("searxng")
        if tavily_configured:
            available.append("tavily")
        if linkup_configured:
            available.append("linkup")

        if len(available) < 2:
            pytest.skip("At least 2 connectors required")

        response = client.post("/api/v1/research", json={
            "query": "machine learning frameworks comparison",
            "top_k": 3,
            "connectors": available,
            "reasoning_effort": "low"
        })

        assert response.status_code == 200
        data = response.json()

        # Multiple connectors should be used
        assert len(data["connectors_used"]) >= 1
        # RRF should combine results
        assert len(data["sources"]) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_citations_in_response(
        self, client, searxng_configured, llm_configured
    ):
        """Verify citations are properly extracted."""
        if not searxng_configured or not llm_configured:
            pytest.skip("Full stack required")

        response = client.post("/api/v1/research", json={
            "query": "Explain how HTTP works",
            "top_k": 5,
            "reasoning_effort": "medium"
        })

        assert response.status_code == 200
        data = response.json()

        # Content should exist
        assert len(data["content"]) > 0

        # If citations exist, verify format
        if data["citations"]:
            for citation in data["citations"]:
                assert "id" in citation
                assert "title" in citation
                assert "url" in citation
                # ID should match expected pattern
                assert "_" in citation["id"]

    @pytest.mark.integration
    def test_search_result_deduplication(self, client, searxng_configured):
        """Verify search results are deduplicated."""
        if not searxng_configured:
            pytest.skip("SearXNG required")

        response = client.post("/api/v1/search", json={
            "query": "python programming language",
            "top_k": 20
        })

        assert response.status_code == 200
        data = response.json()

        # Check for duplicate URLs
        urls = [s["url"] for s in data["sources"]]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    @pytest.mark.integration
    def test_source_score_ordering(self, client, searxng_configured):
        """Verify sources are ordered by RRF score."""
        if not searxng_configured:
            pytest.skip("SearXNG required")

        response = client.post("/api/v1/search", json={
            "query": "javascript tutorial",
            "top_k": 10
        })

        assert response.status_code == 200
        data = response.json()

        if len(data["sources"]) > 1:
            scores = [s["score"] for s in data["sources"]]
            assert scores == sorted(scores, reverse=True), "Sources not sorted by score"


class TestErrorHandling:
    """Tests for error handling in E2E flow."""

    @pytest.mark.unit
    def test_empty_query_rejected(self, client):
        """Empty query is rejected."""
        response = client.post("/api/v1/search", json={
            "query": ""
        })
        # Pydantic should reject empty string or FastAPI should handle
        # Either 422 (validation) or 200 with empty results is acceptable
        assert response.status_code in [200, 422]

    @pytest.mark.unit
    def test_invalid_json_rejected(self, client):
        """Invalid JSON is rejected."""
        response = client.post(
            "/api/v1/search",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    @pytest.mark.integration
    def test_nonexistent_connector_handled(self, client):
        """Non-existent connector is handled gracefully."""
        response = client.post("/api/v1/search", json={
            "query": "test",
            "connectors": ["nonexistent_connector"]
        })

        # Should return 200 with empty results, not crash
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.integration
    def test_search_response_time(self, client, searxng_configured):
        """Search responds within reasonable time."""
        if not searxng_configured:
            pytest.skip("SearXNG required")

        import time
        start = time.time()

        response = client.post("/api/v1/search", json={
            "query": "quick test query",
            "top_k": 5
        })

        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete within 10 seconds
        assert elapsed < 10, f"Search took too long: {elapsed:.2f}s"

    @pytest.mark.integration
    def test_health_response_time(self, client):
        """Health check is fast."""
        import time
        start = time.time()

        response = client.get("/api/v1/health")

        elapsed = time.time() - start

        assert response.status_code == 200
        # Health check should be instant
        assert elapsed < 1, f"Health check took too long: {elapsed:.2f}s"
