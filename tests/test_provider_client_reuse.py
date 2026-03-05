"""Test that provider clients are reused across calls, reducing TCP connection overhead."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from agent_trace.judge.providers import OpenAIJudgeProvider, AnthropicJudgeProvider


class TestOpenAIProviderClientReuse:
    """Tests for OpenAIJudgeProvider client reuse and lifecycle."""

    def test_judge_client_lazy_initialization(self):
        """Verify that httpx.Client is only created when first judge() call is made."""
        provider = OpenAIJudgeProvider(api_key="test-key")
        assert provider._sync_client is None  # Not created yet

        mock_client_instance = Mock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 5/5"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.Client', return_value=mock_client_instance) as mock_client_class:
            # Make first call - client created
            provider.judge("system", "user")
            assert provider._sync_client is not None
            assert mock_client_class.call_count == 1

            # Make second call - client should NOT be recreated
            provider.judge("system", "user2")
            assert mock_client_class.call_count == 1  # Still 1, not recreated

            # Verify post was called twice with the same (non-recreated) client
            assert mock_client_instance.post.call_count == 2

    @pytest.mark.asyncio
    async def test_ajudge_async_client_lazy_initialization(self):
        """Verify that httpx.AsyncClient is only created when first ajudge() call is made."""
        provider = OpenAIJudgeProvider(api_key="test-key")
        assert provider._async_client is None  # Not created yet

        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 5/5"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.AsyncClient', return_value=mock_client_instance) as mock_client_class:
            # Make first call - client created
            await provider.ajudge("system", "user")
            assert provider._async_client is not None
            assert mock_client_class.call_count == 1

            # Make second call - client should NOT be recreated
            await provider.ajudge("system", "user2")
            assert mock_client_class.call_count == 1  # Still 1, not recreated

            # Verify post was called twice with the same (non-recreated) client
            assert mock_client_instance.post.call_count == 2

    def test_close_closes_sync_client(self):
        """Verify that close() closes the httpx.Client if it exists."""
        provider = OpenAIJudgeProvider(api_key="test-key")

        mock_client_instance = Mock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 5/5"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.Client', return_value=mock_client_instance):
            # Create client
            provider.judge("system", "user")
            assert provider._sync_client is not None

            # Close it
            provider.close()

            mock_client_instance.close.assert_called()

    @pytest.mark.asyncio
    async def test_aclose_closes_async_client(self):
        """Verify that aclose() closes the httpx.AsyncClient if it exists."""
        provider = OpenAIJudgeProvider(api_key="test-key")

        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 5/5"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.AsyncClient', return_value=mock_client_instance):
            # Create client
            await provider.ajudge("system", "user")
            assert provider._async_client is not None

            # Close it
            await provider.aclose()

            mock_client_instance.aclose.assert_called()

    def test_context_manager_closes_client(self):
        """Verify that context manager properly closes the sync client."""
        mock_client_instance = Mock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 5/5"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.Client', return_value=mock_client_instance):
            with OpenAIJudgeProvider(api_key="test-key") as provider:
                provider.judge("system", "user")
                assert provider._sync_client is not None

            # After exiting context, client should be closed
            mock_client_instance.close.assert_called()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self):
        """Verify that async context manager properly closes the async client."""
        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Score: 5/5"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.AsyncClient', return_value=mock_client_instance):
            async with OpenAIJudgeProvider(api_key="test-key") as provider:
                await provider.ajudge("system", "user")
                assert provider._async_client is not None

            # After exiting context, client should be closed
            mock_client_instance.aclose.assert_called()


class TestAnthropicProviderClientReuse:
    """Tests for AnthropicJudgeProvider client reuse and lifecycle."""

    def test_judge_client_lazy_initialization(self):
        """Verify that httpx.Client is only created when first judge() call is made."""
        provider = AnthropicJudgeProvider(api_key="test-key")
        assert provider._sync_client is None  # Not created yet

        mock_client_instance = Mock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Score: 5/5"}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.Client', return_value=mock_client_instance) as mock_client_class:
            # Make first call - client created
            provider.judge("system", "user")
            assert provider._sync_client is not None
            assert mock_client_class.call_count == 1

            # Make second call - client should NOT be recreated
            provider.judge("system", "user2")
            assert mock_client_class.call_count == 1  # Still 1, not recreated

            # Verify post was called twice with the same (non-recreated) client
            assert mock_client_instance.post.call_count == 2

    @pytest.mark.asyncio
    async def test_ajudge_async_client_lazy_initialization(self):
        """Verify that httpx.AsyncClient is only created when first ajudge() call is made."""
        provider = AnthropicJudgeProvider(api_key="test-key")
        assert provider._async_client is None  # Not created yet

        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Score: 5/5"}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.AsyncClient', return_value=mock_client_instance) as mock_client_class:
            # Make first call - client created
            await provider.ajudge("system", "user")
            assert provider._async_client is not None
            assert mock_client_class.call_count == 1

            # Make second call - client should NOT be recreated
            await provider.ajudge("system", "user2")
            assert mock_client_class.call_count == 1  # Still 1, not recreated

            # Verify post was called twice with the same (non-recreated) client
            assert mock_client_instance.post.call_count == 2

    def test_context_manager_closes_client(self):
        """Verify that context manager properly closes the sync client."""
        mock_client_instance = Mock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Score: 5/5"}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.Client', return_value=mock_client_instance):
            with AnthropicJudgeProvider(api_key="test-key") as provider:
                provider.judge("system", "user")
                assert provider._sync_client is not None

            # After exiting context, client should be closed
            mock_client_instance.close.assert_called()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self):
        """Verify that async context manager properly closes the async client."""
        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Score: 5/5"}]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch('agent_trace.judge.providers.httpx.AsyncClient', return_value=mock_client_instance):
            async with AnthropicJudgeProvider(api_key="test-key") as provider:
                await provider.ajudge("system", "user")
                assert provider._async_client is not None

            # After exiting context, client should be closed
            mock_client_instance.aclose.assert_called()
