"""
Centralized Model Caller for DeepThinker 2.0.

Provides a unified interface for all HTTP calls to Ollama with:
- Context-managed HTTP clients (no resource leaks)
- Exponential backoff retry logic
- Strict timeout enforcement
- Resource monitoring and cleanup utilities
- Structured logging for debugging

Rules enforced:
1. Never reuse HTTP clients across threads
2. Every call uses a new context-managed client
3. Always cleanup after calls (even on exceptions)
4. All councils/modules must use this module for Ollama calls
"""

import gc
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default Ollama base URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Socket count threshold for warning
SOCKET_WARNING_THRESHOLD = 100


class ModelInvocationError(Exception):
    """Exception raised when model invocation fails after retries."""
    
    def __init__(self, message: str, model: str = "", cause: Optional[Exception] = None):
        super().__init__(message)
        self.model = model
        self.cause = cause


def count_open_sockets() -> int:
    """
    Count the number of open network sockets for the current process.
    
    Uses psutil to inspect the current process's connections.
    Logs a warning if socket count exceeds threshold.
    
    Returns:
        Number of open inet sockets, or -1 if psutil is not available
    """
    try:
        import psutil
        proc = psutil.Process()
        connections = proc.connections(kind='inet')
        count = len(connections)
        
        if count > SOCKET_WARNING_THRESHOLD:
            logger.warning(f"[Resource] High number of open sockets: {count}")
        
        return count
    except ImportError:
        logger.debug("[Resource] psutil not available for socket monitoring")
        return -1
    except Exception as e:
        logger.debug(f"[Resource] Failed to count sockets: {e}")
        return -1


def cleanup_resources() -> None:
    """
    Perform garbage collection to clean up unreferenced resources.
    
    Should be called:
    - After model invocation failures
    - At the end of each mission iteration
    - When high socket count is detected
    """
    gc.collect()
    logger.debug("[Resource] Garbage collection completed")


def call_model(
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    base_url: str = DEFAULT_OLLAMA_URL,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Make a synchronous call to an Ollama model.
    
    Uses httpx.Client with a context manager to ensure proper cleanup.
    Creates a new client for each call (no reuse across invocations).
    
    Args:
        model: Name of the Ollama model to use
        prompt: The prompt to send to the model
        options: Optional model options dictionary. Currently supports:
                - temperature: Sampling temperature (0.0-2.0)
                - NOTE (Phase 3.4): Future hook for Ollama server control options:
                  The options dict can accept num_gpu, num_thread, use_mlock, etc.
                  for future Ollama server-level model configuration. These are NOT
                  currently implemented - DeepThinker delegates model loading/unloading
                  to the Ollama server. This hook exists for potential future use when
                  we need to pass layer offloading or GPU allocation hints to Ollama.
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        base_url: Ollama server base URL
        stream: Whether to stream the response (currently not supported)
        
    Returns:
        Dictionary containing the response from Ollama
        
    Raises:
        ModelInvocationError: If all retries fail
    """
    if options is None:
        options = {}
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }
    
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"[ModelCaller] Opening HTTP client (attempt {attempt + 1}/{max_retries})")
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"[ModelCaller] Closing HTTP client (success)")
                logger.debug(f"[ModelCaller] Sockets open: {count_open_sockets()}")
                
                return result
                
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"[ModelCaller] Timeout calling {model} (attempt {attempt + 1}): {e}")
            
        except httpx.HTTPStatusError as e:
            last_error = e
            logger.warning(f"[ModelCaller] HTTP error calling {model} (attempt {attempt + 1}): {e}")
            
        except httpx.RequestError as e:
            last_error = e
            logger.warning(f"[ModelCaller] Request error calling {model} (attempt {attempt + 1}): {e}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"[ModelCaller] Unexpected error calling {model} (attempt {attempt + 1}): {e}")
        
        finally:
            logger.debug(f"[ModelCaller] Closing HTTP client (attempt {attempt + 1})")
        
        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            backoff = 2 ** attempt
            logger.info(f"[ModelCaller] Retrying in {backoff}s...")
            time.sleep(backoff)
    
    # All retries failed
    cleanup_resources()
    raise ModelInvocationError(
        f"Failed to call model {model} after {max_retries} attempts",
        model=model,
        cause=last_error,
    )


async def call_model_async(
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    base_url: str = DEFAULT_OLLAMA_URL,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Make an asynchronous call to an Ollama model.
    
    Uses httpx.AsyncClient with a context manager to ensure proper cleanup.
    Creates a new client for each call (no reuse across invocations).
    
    Args:
        model: Name of the Ollama model to use
        prompt: The prompt to send to the model
        options: Optional model options (temperature, num_gpu, etc.)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        base_url: Ollama server base URL
        stream: Whether to stream the response (currently not supported)
        
    Returns:
        Dictionary containing the response from Ollama
        
    Raises:
        ModelInvocationError: If all retries fail
    """
    import asyncio
    
    if options is None:
        options = {}
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }
    
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"[ModelCaller] Opening async HTTP client (attempt {attempt + 1}/{max_retries})")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"[ModelCaller] Closing async HTTP client (success)")
                logger.debug(f"[ModelCaller] Sockets open: {count_open_sockets()}")
                
                return result
                
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async timeout calling {model} (attempt {attempt + 1}): {e}")
            
        except httpx.HTTPStatusError as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async HTTP error calling {model} (attempt {attempt + 1}): {e}")
            
        except httpx.RequestError as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async request error calling {model} (attempt {attempt + 1}): {e}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async unexpected error calling {model} (attempt {attempt + 1}): {e}")
        
        finally:
            logger.debug(f"[ModelCaller] Closing async HTTP client (attempt {attempt + 1})")
        
        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            backoff = 2 ** attempt
            logger.info(f"[ModelCaller] Async retrying in {backoff}s...")
            await asyncio.sleep(backoff)
    
    # All retries failed
    cleanup_resources()
    raise ModelInvocationError(
        f"Failed to call model {model} after {max_retries} attempts (async)",
        model=model,
        cause=last_error,
    )


def call_embeddings(
    text: str,
    model: str = "qwen3-embedding:4b",
    timeout: float = 60.0,
    max_retries: int = 3,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> List[float]:
    """
    Get embeddings for text from an Ollama embedding model.
    
    Uses httpx.Client with a context manager to ensure proper cleanup.
    Creates a new client for each call (no reuse across invocations).
    
    Args:
        text: Text to embed
        model: Ollama embedding model name
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        base_url: Ollama server base URL
        
    Returns:
        List of floats representing the embedding vector, or empty list on failure
    """
    payload = {
        "model": model,
        "prompt": text,
    }
    
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"[ModelCaller] Opening HTTP client for embeddings (attempt {attempt + 1}/{max_retries})")
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base_url}/api/embeddings",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"[ModelCaller] Closing HTTP client for embeddings (success)")
                
                embedding = result.get("embedding", [])
                return embedding
                
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"[ModelCaller] Timeout getting embeddings from {model} (attempt {attempt + 1}): {e}")
            
        except httpx.HTTPStatusError as e:
            last_error = e
            logger.warning(f"[ModelCaller] HTTP error getting embeddings from {model} (attempt {attempt + 1}): {e}")
            
        except httpx.RequestError as e:
            last_error = e
            logger.warning(f"[ModelCaller] Request error getting embeddings from {model} (attempt {attempt + 1}): {e}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"[ModelCaller] Unexpected error getting embeddings from {model} (attempt {attempt + 1}): {e}")
        
        finally:
            logger.debug(f"[ModelCaller] Closing HTTP client for embeddings (attempt {attempt + 1})")
        
        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            backoff = 2 ** attempt
            logger.debug(f"[ModelCaller] Retrying embeddings in {backoff}s...")
            time.sleep(backoff)
    
    # All retries failed - return empty list (graceful degradation for embeddings)
    logger.error(f"[ModelCaller] Failed to get embeddings from {model} after {max_retries} attempts")
    cleanup_resources()
    return []


async def call_embeddings_async(
    text: str,
    model: str = "qwen3-embedding:4b",
    timeout: float = 60.0,
    max_retries: int = 3,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> List[float]:
    """
    Get embeddings for text from an Ollama embedding model (async version).
    
    Uses httpx.AsyncClient with a context manager to ensure proper cleanup.
    Creates a new client for each call (no reuse across invocations).
    
    Args:
        text: Text to embed
        model: Ollama embedding model name
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        base_url: Ollama server base URL
        
    Returns:
        List of floats representing the embedding vector, or empty list on failure
    """
    import asyncio
    
    payload = {
        "model": model,
        "prompt": text,
    }
    
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"[ModelCaller] Opening async HTTP client for embeddings (attempt {attempt + 1}/{max_retries})")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"[ModelCaller] Closing async HTTP client for embeddings (success)")
                
                embedding = result.get("embedding", [])
                return embedding
                
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async timeout getting embeddings from {model} (attempt {attempt + 1}): {e}")
            
        except httpx.HTTPStatusError as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async HTTP error getting embeddings from {model} (attempt {attempt + 1}): {e}")
            
        except httpx.RequestError as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async request error getting embeddings from {model} (attempt {attempt + 1}): {e}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"[ModelCaller] Async unexpected error getting embeddings from {model} (attempt {attempt + 1}): {e}")
        
        finally:
            logger.debug(f"[ModelCaller] Closing async HTTP client for embeddings (attempt {attempt + 1})")
        
        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            backoff = 2 ** attempt
            logger.debug(f"[ModelCaller] Async retrying embeddings in {backoff}s...")
            await asyncio.sleep(backoff)
    
    # All retries failed - return empty list (graceful degradation for embeddings)
    logger.error(f"[ModelCaller] Failed to get embeddings from {model} after {max_retries} attempts (async)")
    cleanup_resources()
    return []

