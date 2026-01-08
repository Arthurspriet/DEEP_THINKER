"""
Tests for the centralized model_caller module.

Tests cover:
- Normal call success
- Timeout handling
- Retry logic with exponential backoff
- Context manager properly closes connections
- Socket counting functionality
- Concurrent calls don't leak resources
"""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from deepthinker.models.model_caller import (
    call_model,
    call_model_async,
    call_embeddings,
    call_embeddings_async,
    cleanup_resources,
    count_open_sockets,
    ModelInvocationError,
    SOCKET_WARNING_THRESHOLD,
)


class TestCallModel:
    """Tests for the synchronous call_model function."""
    
    def test_call_model_success(self):
        """Test successful model call returns response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello, world!"}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = call_model(
                model="test-model",
                prompt="Hello",
                timeout=30.0,
                max_retries=1,
            )
            
            assert result == {"response": "Hello, world!"}
            mock_client.post.assert_called_once()
    
    def test_call_model_timeout_raises_error(self):
        """Test that timeout raises ModelInvocationError after retries."""
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(ModelInvocationError) as exc_info:
                call_model(
                    model="test-model",
                    prompt="Hello",
                    timeout=1.0,
                    max_retries=1,  # Only 1 retry to speed up test
                )
            
            assert "test-model" in str(exc_info.value)
            assert exc_info.value.model == "test-model"
    
    def test_call_model_retry_logic(self):
        """Test that retry logic attempts multiple times."""
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RequestError("Connection failed")
            # Success on third attempt
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Success after retry"}
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = side_effect
            mock_client_class.return_value = mock_client
            
            with patch('time.sleep'):  # Speed up test
                result = call_model(
                    model="test-model",
                    prompt="Hello",
                    max_retries=3,
                )
            
            assert result == {"response": "Success after retry"}
            assert call_count == 3
    
    def test_call_model_context_manager_closes_client(self):
        """Test that HTTP client is properly closed via context manager."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test"}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            call_model(model="test", prompt="test", max_retries=1)
            
            # Verify context manager was used
            mock_client.__enter__.assert_called_once()
            mock_client.__exit__.assert_called_once()


class TestCallEmbeddings:
    """Tests for the call_embeddings function."""
    
    def test_call_embeddings_success(self):
        """Test successful embedding call returns vector."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = call_embeddings(
                text="Hello",
                model="test-embedding",
                max_retries=1,
            )
            
            assert result == [0.1, 0.2, 0.3]
    
    def test_call_embeddings_failure_returns_empty_list(self):
        """Test that embedding failures return empty list (graceful degradation)."""
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.RequestError("Failed")
            mock_client_class.return_value = mock_client
            
            with patch('time.sleep'):  # Speed up test
                result = call_embeddings(
                    text="Hello",
                    max_retries=1,
                )
            
            # Should return empty list, not raise exception
            assert result == []


class TestResourceManagement:
    """Tests for resource management functions."""
    
    def test_count_open_sockets_returns_int(self):
        """Test socket counting returns an integer."""
        # This might return -1 if psutil is not available
        count = count_open_sockets()
        assert isinstance(count, int)
    
    def test_count_open_sockets_with_psutil_mock(self):
        """Test socket counting with mocked psutil."""
        mock_proc = MagicMock()
        mock_proc.connections.return_value = [1, 2, 3]  # 3 mock connections
        
        with patch('psutil.Process', return_value=mock_proc):
            count = count_open_sockets()
            assert count == 3
    
    def test_count_open_sockets_warning_threshold(self):
        """Test warning is logged when socket count exceeds threshold."""
        mock_proc = MagicMock()
        # Create mock connections exceeding threshold
        mock_proc.connections.return_value = list(range(SOCKET_WARNING_THRESHOLD + 10))
        
        with patch('psutil.Process', return_value=mock_proc):
            with patch('deepthinker.models.model_caller.logger') as mock_logger:
                count = count_open_sockets()
                assert count > SOCKET_WARNING_THRESHOLD
                mock_logger.warning.assert_called()
    
    def test_cleanup_resources_runs_gc(self):
        """Test that cleanup_resources calls garbage collection."""
        with patch('gc.collect') as mock_gc:
            cleanup_resources()
            mock_gc.assert_called_once()


class AsyncContextManager:
    """Helper class for mocking async context managers."""
    
    def __init__(self, mock_client, mock_response):
        self.mock_client = mock_client
        self.mock_response = mock_response
    
    async def __aenter__(self):
        return self.mock_client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class TestAsyncFunctions:
    """Tests for async versions of model caller functions."""
    
    @pytest.mark.asyncio
    async def test_call_model_async_success(self):
        """Test async model call returns response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Async hello!"}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        
        # Make post return an awaitable
        async def mock_post(*args, **kwargs):
            return mock_response
        mock_client.post = mock_post
        
        # Use helper class for async context manager
        async_cm = AsyncContextManager(mock_client, mock_response)
        
        with patch('httpx.AsyncClient', return_value=async_cm):
            result = await call_model_async(
                model="test-model",
                prompt="Hello",
                max_retries=1,
            )
            
            assert result == {"response": "Async hello!"}
    
    @pytest.mark.asyncio
    async def test_call_embeddings_async_success(self):
        """Test async embedding call returns vector."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        
        async def mock_post(*args, **kwargs):
            return mock_response
        mock_client.post = mock_post
        
        async_cm = AsyncContextManager(mock_client, mock_response)
        
        with patch('httpx.AsyncClient', return_value=async_cm):
            result = await call_embeddings_async(
                text="Hello",
                max_retries=1,
            )
            
            assert result == [0.4, 0.5, 0.6]


class TestModelInvocationError:
    """Tests for the ModelInvocationError exception."""
    
    def test_error_contains_model_name(self):
        """Test that error contains model name."""
        error = ModelInvocationError("Test error", model="my-model")
        assert error.model == "my-model"
        assert "Test error" in str(error)
    
    def test_error_stores_cause(self):
        """Test that error stores the cause exception."""
        cause = ValueError("Original error")
        error = ModelInvocationError("Wrapper", model="test", cause=cause)
        assert error.cause == cause


class TestConcurrentCalls:
    """Tests for concurrent call behavior."""
    
    def test_concurrent_calls_use_separate_clients(self):
        """Test that concurrent calls create separate HTTP clients."""
        client_instances = []
        
        def track_client(*args, **kwargs):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "ok"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            client_instances.append(mock_client)
            return mock_client
        
        with patch('httpx.Client', side_effect=track_client):
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(call_model, "test", f"prompt{i}", None, 10.0, 1)
                    for i in range(3)
                ]
                for f in futures:
                    f.result()
            
            # Each call should have created a separate client
            assert len(client_instances) == 3
            # Each client should have been closed
            for client in client_instances:
                client.__exit__.assert_called()

