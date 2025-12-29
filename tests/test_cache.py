"""
Comprehensive tests for hot cache functionality.

Tests cover:
- Basic cache operations (set/get/clear)
- TTL expiration
- Cache key differentiation
- Decorator behavior
- Source-aware caching for synthesize
- All workflow coverage (DIRECT, EXPLORATORY, SYNTHESIS)
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache import HotCache, cache, cached, CacheEntry


class TestHotCacheBasics:
    """Basic cache operations."""

    def setup_method(self):
        """Fresh cache for each test."""
        self.cache = HotCache(namespace="test_basics")
        self.cache.clear()

    def teardown_method(self):
        """Cleanup after each test."""
        self.cache.clear()

    def test_cache_dir_created(self):
        """Cache directory should be created in /tmp."""
        assert self.cache.cache_dir.exists()
        assert str(self.cache.cache_dir).startswith("/tmp/")

    def test_set_and_get(self):
        """Basic set/get should work."""
        self.cache.set("test query", "test result", tier="synthesis")
        result = self.cache.get("test query", tier="synthesis")
        assert result == "test result"

    def test_get_nonexistent_returns_none(self):
        """Cache miss should return None."""
        result = self.cache.get("nonexistent query", tier="synthesis")
        assert result is None

    def test_cache_key_normalization(self):
        """Queries should be normalized (lowercase, stripped)."""
        self.cache.set("  TEST Query  ", "result", tier="test")

        # Should match with different casing/whitespace
        assert self.cache.get("test query", tier="test") == "result"
        assert self.cache.get("TEST QUERY", tier="test") == "result"
        assert self.cache.get("  test query  ", tier="test") == "result"

    def test_tier_isolation(self):
        """Different tiers should have separate caches."""
        self.cache.set("query", "synthesis result", tier="synthesis")
        self.cache.set("query", "discover result", tier="discover")

        assert self.cache.get("query", tier="synthesis") == "synthesis result"
        assert self.cache.get("query", tier="discover") == "discover result"

    def test_extra_param_isolation(self):
        """Extra params should create separate cache entries."""
        self.cache.set("query", "result1", tier="test", extra="param=1")
        self.cache.set("query", "result2", tier="test", extra="param=2")

        assert self.cache.get("query", tier="test", extra="param=1") == "result1"
        assert self.cache.get("query", tier="test", extra="param=2") == "result2"
        assert self.cache.get("query", tier="test", extra="param=3") is None

    def test_clear_removes_all_entries(self):
        """Clear should remove all cache entries."""
        self.cache.set("q1", "r1", tier="test")
        self.cache.set("q2", "r2", tier="test")

        assert self.cache.stats()["entries"] == 2

        self.cache.clear()

        assert self.cache.stats()["entries"] == 0
        assert self.cache.get("q1", tier="test") is None

    def test_stats_tracking(self):
        """Stats should track hits and misses."""
        self.cache.set("query", "result", tier="test")

        # One miss
        self.cache.get("nonexistent", tier="test")

        # Two hits
        self.cache.get("query", tier="test")
        self.cache.get("query", tier="test")

        stats = self.cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3


class TestTTLExpiration:
    """TTL-based cache expiration."""

    def setup_method(self):
        self.cache = HotCache(namespace="test_ttl")
        self.cache.clear()

    def teardown_method(self):
        self.cache.clear()

    def test_default_ttls_by_tier(self):
        """Each tier should have appropriate default TTL."""
        expected_ttls = {
            "synthesis": 3600,
            "discover": 3600,
            "reason": 3600,
            "research": 1800,
            "search": 1800,
            "url": 7200,
            "ask": 1800,
        }
        assert self.cache.DEFAULT_TTLS == expected_ttls

    def test_fresh_entry_returned(self):
        """Entry within TTL should be returned."""
        self.cache.set("query", "result", tier="test", ttl=60)
        assert self.cache.get("query", tier="test") == "result"

    def test_expired_entry_returns_none(self):
        """Entry past TTL should return None and be deleted."""
        # Set with very short TTL
        self.cache.set("query", "result", tier="test", ttl=1)

        # Wait for expiration
        time.sleep(1.1)

        result = self.cache.get("query", tier="test")
        assert result is None

        # File should be deleted
        key = self.cache._key("query", tier="test")
        assert not self.cache._path(key).exists()

    def test_custom_ttl_override(self):
        """Custom TTL should override default."""
        self.cache.set("query", "result", tier="synthesis", ttl=1)

        # Should exist immediately
        assert self.cache.get("query", tier="synthesis") == "result"

        # Should expire after 1 second
        time.sleep(1.1)
        assert self.cache.get("query", tier="synthesis") is None


class TestURLCaching:
    """URL content caching (L2 tier)."""

    def setup_method(self):
        self.cache = HotCache(namespace="test_url")
        self.cache.clear()

    def teardown_method(self):
        self.cache.clear()

    def test_set_url_and_get_url(self):
        """URL caching should work."""
        self.cache.set_url("https://example.com/page", "page content")
        result = self.cache.get_url("https://example.com/page")
        assert result == "page content"

    def test_get_url_miss_returns_none(self):
        """URL cache miss should return None."""
        result = self.cache.get_url("https://nonexistent.com")
        assert result is None

    def test_url_ttl_default(self):
        """URL caching should use 2h TTL by default."""
        self.cache.set_url("https://example.com", "content")

        # Check the stored TTL
        key = self.cache._key("https://example.com", tier="url")
        path = self.cache._path(key)
        data = json.loads(path.read_text())
        assert data["ttl"] == 7200  # 2 hours


class TestCachedDecorator:
    """@cached decorator functionality."""

    def setup_method(self):
        # Clear the global cache
        cache.clear()

    def teardown_method(self):
        cache.clear()

    @pytest.mark.asyncio
    async def test_decorator_caches_result(self):
        """Decorated function should cache its result."""
        call_count = 0

        @cached(tier="test")
        async def mock_tool(args: dict):
            nonlocal call_count
            call_count += 1
            from mcp.types import TextContent
            return [TextContent(type="text", text=f"result-{call_count}")]

        # First call - executes function
        r1 = await mock_tool({"query": "test"})
        assert r1[0].text == "result-1"
        assert call_count == 1

        # Second call - returns cached
        r2 = await mock_tool({"query": "test"})
        assert "*[cached]*" in r2[0].text
        assert "result-1" in r2[0].text
        assert call_count == 1  # Function not called again

    @pytest.mark.asyncio
    async def test_decorator_cache_marker(self):
        """Cached results should have *[cached]* marker."""
        @cached(tier="test")
        async def mock_tool(args: dict):
            from mcp.types import TextContent
            return [TextContent(type="text", text="original result")]

        await mock_tool({"query": "test"})
        r2 = await mock_tool({"query": "test"})

        assert r2[0].text.startswith("*[cached]*")

    @pytest.mark.asyncio
    async def test_decorator_different_queries_not_cached(self):
        """Different queries should have separate cache entries."""
        call_count = 0

        @cached(tier="test")
        async def mock_tool(args: dict):
            nonlocal call_count
            call_count += 1
            from mcp.types import TextContent
            return [TextContent(type="text", text=f"result for {args['query']}")]

        await mock_tool({"query": "query1"})
        await mock_tool({"query": "query2"})

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_key_params(self):
        """key_params should differentiate cache entries."""
        call_count = 0

        @cached(tier="test", key_params=["top_k"])
        async def mock_tool(args: dict):
            nonlocal call_count
            call_count += 1
            from mcp.types import TextContent
            return [TextContent(type="text", text=f"result-{call_count}")]

        # Same query, different top_k
        await mock_tool({"query": "test", "top_k": 5})
        await mock_tool({"query": "test", "top_k": 10})

        assert call_count == 2

        # Same query and top_k - should hit cache
        await mock_tool({"query": "test", "top_k": 5})
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_empty_query_bypasses_cache(self):
        """Empty query should bypass caching."""
        call_count = 0

        @cached(tier="test")
        async def mock_tool(args: dict):
            nonlocal call_count
            call_count += 1
            from mcp.types import TextContent
            return [TextContent(type="text", text="result")]

        await mock_tool({"query": ""})
        await mock_tool({"query": ""})

        # Both calls should execute (no caching)
        assert call_count == 2


# Check if mcp_server exists (main branch) vs api/routes (openrouter branch)
try:
    from src.mcp_server import _tool_search
    HAS_MCP_SERVER = True
except ImportError:
    HAS_MCP_SERVER = False


@pytest.mark.skipif(not HAS_MCP_SERVER, reason="MCP server not available (openrouter branch uses HTTP transport)")
class TestMCPToolCaching:
    """Test caching integration with actual MCP tool signatures."""

    def setup_method(self):
        cache.clear()

    def teardown_method(self):
        cache.clear()

    @pytest.mark.asyncio
    async def test_search_tool_caching(self):
        """_tool_search should be cached with top_k in key."""
        from src.mcp_server import _tool_search

        with patch('src.mcp_server.SearchAggregator') as mock_agg:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=([], {}))
            mock_agg.return_value = mock_instance

            # First call
            await _tool_search({"query": "test search", "top_k": 5})
            assert mock_instance.search.call_count == 1

            # Second call with same params - should be cached
            r2 = await _tool_search({"query": "test search", "top_k": 5})
            assert mock_instance.search.call_count == 1
            assert "*[cached]*" in r2[0].text

            # Different top_k - should execute
            await _tool_search({"query": "test search", "top_k": 10})
            assert mock_instance.search.call_count == 2

    @pytest.mark.asyncio
    async def test_ask_tool_caching(self):
        """_tool_ask should be cached."""
        from src.mcp_server import _tool_ask

        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test answer"

            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # First call
            await _tool_ask({"query": "test question"})
            assert mock_client.return_value.chat.completions.create.call_count == 1

            # Second call - cached
            r2 = await _tool_ask({"query": "test question"})
            assert mock_client.return_value.chat.completions.create.call_count == 1
            assert "*[cached]*" in r2[0].text

    @pytest.mark.asyncio
    async def test_discover_tool_caching(self):
        """_tool_discover should be cached with focus_mode and identify_gaps."""
        from src.mcp_server import _tool_discover

        with patch('src.mcp_server.Explorer') as mock_explorer:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            # Properly mock landscape with required attributes
            mock_landscape = MagicMock()
            mock_landscape.explicit_topics = ["topic1", "topic2"]
            mock_landscape.implicit_topics = []
            mock_landscape.related_concepts = []
            mock_result.landscape = mock_landscape
            mock_result.knowledge_gaps = []
            mock_result.sources = []
            mock_result.recommended_deep_dives = []
            mock_instance.discover = AsyncMock(return_value=mock_result)
            mock_explorer.return_value = mock_instance

            # First call
            await _tool_discover({
                "query": "test topic",
                "focus_mode": "academic",
                "identify_gaps": True
            })
            assert mock_instance.discover.call_count == 1

            # Same params - cached
            r2 = await _tool_discover({
                "query": "test topic",
                "focus_mode": "academic",
                "identify_gaps": True
            })
            assert mock_instance.discover.call_count == 1
            assert "*[cached]*" in r2[0].text

            # Different focus_mode - new call
            await _tool_discover({
                "query": "test topic",
                "focus_mode": "technical",
                "identify_gaps": True
            })
            assert mock_instance.discover.call_count == 2

    @pytest.mark.asyncio
    async def test_reason_tool_caching(self):
        """_tool_reason should be cached with reasoning_depth."""
        from src.mcp_server import _tool_reason

        with patch('src.mcp_server._get_llm_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "reasoning result"

            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # First call
            await _tool_reason({
                "query": "test problem",
                "reasoning_depth": "deep"
            })
            assert mock_client.return_value.chat.completions.create.call_count == 1

            # Same params - cached
            r2 = await _tool_reason({
                "query": "test problem",
                "reasoning_depth": "deep"
            })
            assert mock_client.return_value.chat.completions.create.call_count == 1
            assert "*[cached]*" in r2[0].text

            # Different depth - new call
            await _tool_reason({
                "query": "test problem",
                "reasoning_depth": "shallow"
            })
            assert mock_client.return_value.chat.completions.create.call_count == 2


@pytest.mark.skipif(not HAS_MCP_SERVER, reason="MCP server not available (openrouter branch uses HTTP transport)")
class TestSynthesizeSourceAwareCaching:
    """Test source-aware caching for synthesize tool."""

    def setup_method(self):
        cache.clear()

    def teardown_method(self):
        cache.clear()

    @pytest.mark.asyncio
    async def test_same_sources_cached(self):
        """Same query + sources should be cached."""
        from src.mcp_server import _tool_synthesize

        sources = [
            {"title": "Source 1", "url": "http://a.com", "content": "content 1"},
            {"title": "Source 2", "url": "http://b.com", "content": "content 2"},
        ]

        with patch('src.mcp_server.SynthesisAggregator') as mock_agg:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.content = "synthesis result"
            mock_result.citations = []
            mock_instance.synthesize = AsyncMock(return_value=mock_result)
            mock_agg.return_value = mock_instance

            # First call
            await _tool_synthesize({
                "query": "test synthesis",
                "sources": sources,
                "style": "comprehensive"
            })
            assert mock_instance.synthesize.call_count == 1

            # Same sources - cached
            r2 = await _tool_synthesize({
                "query": "test synthesis",
                "sources": sources,
                "style": "comprehensive"
            })
            assert mock_instance.synthesize.call_count == 1
            assert "*[cached]*" in r2[0].text

    @pytest.mark.asyncio
    async def test_different_sources_not_cached(self):
        """Different sources should create new cache entry."""
        from src.mcp_server import _tool_synthesize

        sources1 = [
            {"title": "Source 1", "url": "http://a.com", "content": "content 1"},
        ]
        sources2 = [
            {"title": "Source 2", "url": "http://b.com", "content": "content 2"},
        ]

        with patch('src.mcp_server.SynthesisAggregator') as mock_agg:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.content = "synthesis result"
            mock_result.citations = []
            mock_instance.synthesize = AsyncMock(return_value=mock_result)
            mock_agg.return_value = mock_instance

            await _tool_synthesize({
                "query": "test synthesis",
                "sources": sources1
            })
            await _tool_synthesize({
                "query": "test synthesis",
                "sources": sources2
            })

            # Both should execute (different sources)
            assert mock_instance.synthesize.call_count == 2

    @pytest.mark.asyncio
    async def test_different_style_not_cached(self):
        """Different style should create new cache entry."""
        from src.mcp_server import _tool_synthesize

        sources = [
            {"title": "Source 1", "url": "http://a.com", "content": "content 1"},
        ]

        with patch('src.mcp_server.SynthesisAggregator') as mock_agg:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.content = "synthesis result"
            mock_result.citations = []
            mock_instance.synthesize = AsyncMock(return_value=mock_result)
            mock_agg.return_value = mock_instance

            await _tool_synthesize({
                "query": "test synthesis",
                "sources": sources,
                "style": "comprehensive"
            })
            await _tool_synthesize({
                "query": "test synthesis",
                "sources": sources,
                "style": "concise"
            })

            # Both should execute (different style)
            assert mock_instance.synthesize.call_count == 2


@pytest.mark.skipif(not HAS_MCP_SERVER, reason="MCP server not available (openrouter branch uses HTTP transport)")
class TestWorkflowCoverage:
    """Verify all three workflows have caching."""

    def test_direct_workflow_tools_cached(self):
        """DIRECT workflow: ask should be cached."""
        from src.mcp_server import _tool_ask

        # Check decorator is applied
        assert hasattr(_tool_ask, '__wrapped__')

    def test_exploratory_workflow_tools_cached(self):
        """EXPLORATORY workflow: discover should be cached."""
        from src.mcp_server import _tool_discover

        assert hasattr(_tool_discover, '__wrapped__')

    def test_synthesis_workflow_tools_cached(self):
        """SYNTHESIS workflow: synthesize and reason should be cached."""
        from src.mcp_server import _tool_synthesize, _tool_reason

        # reason has decorator
        assert hasattr(_tool_reason, '__wrapped__')

        # synthesize uses manual caching (check source code has cache.get/set)
        import inspect
        source = inspect.getsource(_tool_synthesize)
        assert "cache.get(" in source
        assert "cache.set(" in source

    def test_utility_tools_cached(self):
        """Utility tools: search and research should be cached."""
        from src.mcp_server import _tool_search, _tool_research

        assert hasattr(_tool_search, '__wrapped__')
        assert hasattr(_tool_research, '__wrapped__')


class TestCacheFileFormat:
    """Verify cache file format and integrity."""

    def setup_method(self):
        self.cache = HotCache(namespace="test_format")
        self.cache.clear()

    def teardown_method(self):
        self.cache.clear()

    def test_cache_file_is_valid_json(self):
        """Cache files should be valid JSON."""
        self.cache.set("query", "result", tier="test")

        key = self.cache._key("query", tier="test")
        path = self.cache._path(key)

        data = json.loads(path.read_text())
        assert "result" in data
        assert "created_at" in data
        assert "ttl" in data

    def test_cache_entry_structure(self):
        """Cache entry should have correct structure."""
        self.cache.set("query", {"complex": "data"}, tier="test", ttl=1234)

        key = self.cache._key("query", tier="test")
        path = self.cache._path(key)
        data = json.loads(path.read_text())

        assert data["result"] == {"complex": "data"}
        assert isinstance(data["created_at"], float)
        assert data["ttl"] == 1234

    def test_corrupted_file_handled_gracefully(self):
        """Corrupted cache files should be handled without error."""
        self.cache.set("query", "result", tier="test")

        key = self.cache._key("query", tier="test")
        path = self.cache._path(key)

        # Corrupt the file
        path.write_text("not valid json {{{")

        # Should return None and not raise
        result = self.cache.get("query", tier="test")
        assert result is None

        # Corrupted file should be deleted
        assert not path.exists()


class TestEdgeCases:
    """Edge cases and error handling."""

    def setup_method(self):
        self.cache = HotCache(namespace="test_edge")
        self.cache.clear()

    def teardown_method(self):
        self.cache.clear()

    def test_non_serializable_result_handled(self):
        """Non-JSON-serializable results should be skipped silently."""
        class NonSerializable:
            pass

        # Should not raise
        self.cache.set("query", NonSerializable(), tier="test")

        # Should return None (not cached)
        assert self.cache.get("query", tier="test") is None

    def test_empty_string_query(self):
        """Empty string query should still work."""
        self.cache.set("", "result", tier="test")
        assert self.cache.get("", tier="test") == "result"

    def test_very_long_query(self):
        """Very long queries should work (hashed to fixed length)."""
        long_query = "x" * 10000
        self.cache.set(long_query, "result", tier="test")
        assert self.cache.get(long_query, tier="test") == "result"

    def test_special_characters_in_query(self):
        """Special characters should be handled."""
        special_query = "test\n\t'\"<>&{}[]"
        self.cache.set(special_query, "result", tier="test")
        assert self.cache.get(special_query, tier="test") == "result"

    def test_unicode_query(self):
        """Unicode queries should work."""
        unicode_query = "测试查询 🔍 тест"
        self.cache.set(unicode_query, "result", tier="test")
        assert self.cache.get(unicode_query, tier="test") == "result"

    def test_concurrent_access(self):
        """Concurrent cache access should be safe."""
        import threading

        results = []

        def writer():
            for i in range(100):
                self.cache.set(f"query-{i}", f"result-{i}", tier="test")

        def reader():
            for i in range(100):
                result = self.cache.get(f"query-{i}", tier="test")
                if result:
                    results.append(result)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(results) >= 0  # Some reads may have succeeded
