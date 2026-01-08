"""
LiteLLM Monitoring and Observability Module

Provides comprehensive monitoring for all LLM calls including:
- Request/response logging
- Cost tracking
- Latency monitoring
- Error tracking
- Token usage statistics
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    from litellm import completion
    from litellm.integrations.custom_logger import CustomLogger
    try:
        from litellm import set_verbose
    except ImportError:
        # set_verbose might not be available in all versions
        set_verbose = None
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    set_verbose = None
    print("⚠️  LiteLLM not available. Monitoring will be disabled.")


class DeepThinkerLiteLLMLogger(CustomLogger):
    """
    Custom LiteLLM logger that tracks all LLM interactions.
    
    Logs requests, responses, costs, latency, and errors to both
    console and file for comprehensive observability.
    """
    
    def __init__(self, log_dir: str = "logs", verbose: bool = False):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            verbose: Whether to print verbose output to console
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Statistics tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.errors = []
        
        # Per-agent statistics tracking
        self.agent_stats = {
            "coder": {"calls": 0, "tokens": 0, "latency": 0, "cost": 0},
            "evaluator": {"calls": 0, "tokens": 0, "latency": 0, "cost": 0},
            "simulator": {"calls": 0, "tokens": 0, "latency": 0, "cost": 0}
        }
        
        # Current agent context (set externally)
        self.current_agent = None
        self.current_iteration = None
        
        # Session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"llm_session_{timestamp}.jsonl"
        
        if self.verbose:
            print(f"📊 LiteLLM Monitoring initialized. Logs: {self.session_log_file}")
    
    def log_pre_api_call(self, model, messages, kwargs):
        """Called before each API call."""
        self.total_requests += 1
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 LLM Request #{self.total_requests}")
            if self.current_agent:
                print(f"   Agent: {self.current_agent}")
            if self.current_iteration:
                print(f"   Iteration: {self.current_iteration}")
            print(f"   Model: {model}")
            print(f"   Messages: {len(messages) if isinstance(messages, list) else 1}")
            print(f"{'='*60}")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        """Called after successful API call."""
        try:
            latency = end_time - start_time
            self.total_latency += latency
            
            # Extract token usage
            usage = getattr(response_obj, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', prompt_tokens + completion_tokens)
                self.total_tokens += total_tokens
            else:
                prompt_tokens = completion_tokens = total_tokens = 0
            
            # Extract cost (LiteLLM calculates this)
            cost = getattr(response_obj, '_hidden_params', {}).get('response_cost', 0.0)
            self.total_cost += cost
            
            # Track per-agent statistics
            if self.current_agent and self.current_agent in self.agent_stats:
                self.agent_stats[self.current_agent]["calls"] += 1
                self.agent_stats[self.current_agent]["tokens"] += total_tokens
                self.agent_stats[self.current_agent]["latency"] += latency
                self.agent_stats[self.current_agent]["cost"] += cost
            
            # Log to file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "model": kwargs.get('model', 'unknown'),
                "agent": self.current_agent,
                "iteration": self.current_iteration,
                "latency_seconds": latency,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens
                },
                "cost_usd": cost,
                "request_id": self.total_requests
            }
            
            self._write_log(log_entry)
            
            if self.verbose:
                print(f"\n✅ LLM Response #{self.total_requests}")
                print(f"   Latency: {latency:.2f}s")
                print(f"   Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
                if cost > 0:
                    print(f"   Cost: ${cost:.6f}")
                print(f"{'='*60}\n")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error logging API call: {e}")
    
    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called when API call fails."""
        try:
            latency = end_time - start_time
            error_msg = str(response_obj)
            
            self.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
                "model": kwargs.get('model', 'unknown')
            })
            
            # Log to file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "model": kwargs.get('model', 'unknown'),
                "latency_seconds": latency,
                "error": error_msg,
                "request_id": self.total_requests
            }
            
            self._write_log(log_entry)
            
            if self.verbose:
                print(f"\n❌ LLM Error #{self.total_requests}")
                print(f"   Model: {kwargs.get('model', 'unknown')}")
                print(f"   Error: {error_msg}")
                print(f"{'='*60}\n")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error logging failure: {e}")
    
    def _write_log(self, entry: Dict[str, Any]):
        """Write log entry to file."""
        try:
            with open(self.session_log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error writing log: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current monitoring statistics.
        
        Returns:
            Dictionary with monitoring stats
        """
        # Calculate per-agent averages
        agent_stats_with_avg = {}
        for agent, stats in self.agent_stats.items():
            agent_stats_with_avg[agent] = {
                "calls": stats["calls"],
                "total_tokens": stats["tokens"],
                "total_latency_seconds": stats["latency"],
                "total_cost_usd": stats["cost"],
                "avg_tokens_per_call": stats["tokens"] / max(stats["calls"], 1),
                "avg_latency_seconds": stats["latency"] / max(stats["calls"], 1),
            }
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "total_latency_seconds": self.total_latency,
            "average_latency_seconds": self.total_latency / max(self.total_requests, 1),
            "average_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
            "errors": len(self.errors),
            "success_rate": (self.total_requests - len(self.errors)) / max(self.total_requests, 1) * 100,
            "per_agent": agent_stats_with_avg
        }
    
    def set_agent_context(self, agent_name: Optional[str], iteration: Optional[int] = None):
        """
        Set the current agent context for tracking.
        
        Args:
            agent_name: Name of the current agent (coder, evaluator, simulator)
            iteration: Current iteration number
        """
        self.current_agent = agent_name
        self.current_iteration = iteration
    
    def print_summary(self):
        """Print a summary of monitoring statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 LiteLLM Monitoring Summary")
        print("="*60)
        print(f"Total Requests:     {stats['total_requests']}")
        print(f"Total Tokens:       {stats['total_tokens']:,}")
        print(f"Total Cost:         ${stats['total_cost_usd']:.6f}")
        print(f"Total Latency:      {stats['total_latency_seconds']:.2f}s")
        print(f"Avg Latency:        {stats['average_latency_seconds']:.2f}s")
        print(f"Avg Tokens/Request: {stats['average_tokens_per_request']:.1f}")
        print(f"Errors:             {stats['errors']}")
        print(f"Success Rate:       {stats['success_rate']:.1f}%")
        print(f"Log File:           {self.session_log_file}")
        print("="*60 + "\n")


class LiteLLMMonitor:
    """
    Thread-safe singleton manager for LiteLLM monitoring.
    
    Provides global access to monitoring functionality and
    configuration for the entire DeepThinker system.
    """
    
    _instance: Optional['LiteLLMMonitor'] = None
    _logger: Optional[DeepThinkerLiteLLMLogger] = None
    _enabled: bool = False
    _lock = None  # Will be initialized to threading.Lock()
    
    def __new__(cls):
        import threading
        
        # Initialize lock on first access
        if cls._lock is None:
            cls._lock = threading.Lock()
        
        # Thread-safe singleton pattern with double-checked locking
        if cls._instance is None:
            with cls._lock:
                # Double-check within lock to prevent race conditions
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(
        cls,
        log_dir: str = "logs",
        verbose: bool = False,
        enable_console_output: bool = False
    ):
        """
        Initialize LiteLLM monitoring.
        
        Args:
            log_dir: Directory for log files
            verbose: Enable verbose logging
            enable_console_output: Enable LiteLLM console output
        """
        if not LITELLM_AVAILABLE:
            print("⚠️  LiteLLM not available. Monitoring disabled.")
            return
        
        instance = cls()
        
        # Create custom logger
        instance._logger = DeepThinkerLiteLLMLogger(log_dir=log_dir, verbose=verbose)
        
        # Configure LiteLLM
        if set_verbose is not None and callable(set_verbose):
            set_verbose(enable_console_output)
        
        # Set environment variables for LiteLLM
        os.environ["LITELLM_LOG"] = "DEBUG" if verbose else "INFO"
        
        instance._enabled = True
        
        if verbose:
            print("✅ LiteLLM Monitoring enabled")
    
    @classmethod
    def get_logger(cls) -> Optional[DeepThinkerLiteLLMLogger]:
        """Get the current logger instance."""
        instance = cls()
        return instance._logger
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if monitoring is enabled."""
        instance = cls()
        return instance._enabled
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        instance = cls()
        if instance._logger:
            return instance._logger.get_stats()
        return {}
    
    @classmethod
    def print_summary(cls):
        """Print monitoring summary."""
        instance = cls()
        if instance._logger:
            instance._logger.print_summary()
    
    @classmethod
    def configure_for_ollama(cls, ollama_api_base: str = "http://localhost:11434"):
        """
        Configure LiteLLM for Ollama integration.
        
        Args:
            ollama_api_base: Base URL for Ollama API
        """
        os.environ["OLLAMA_API_BASE"] = ollama_api_base
        
        if cls.is_enabled() and cls.get_logger() and cls.get_logger().verbose:
            print(f"🔧 LiteLLM configured for Ollama: {ollama_api_base}")


def enable_monitoring(
    log_dir: str = "logs",
    verbose: bool = False,
    enable_console_output: bool = False,
    ollama_api_base: str = "http://localhost:11434"
):
    """
    Convenience function to enable LiteLLM monitoring.
    
    Args:
        log_dir: Directory for log files
        verbose: Enable verbose logging
        enable_console_output: Enable LiteLLM console output
        ollama_api_base: Base URL for Ollama API
    
    Returns:
        The monitoring logger instance
    """
    LiteLLMMonitor.initialize(
        log_dir=log_dir,
        verbose=verbose,
        enable_console_output=enable_console_output
    )
    LiteLLMMonitor.configure_for_ollama(ollama_api_base)
    
    return LiteLLMMonitor.get_logger()


def get_monitoring_stats() -> Dict[str, Any]:
    """
    Get current monitoring statistics.
    
    Returns:
        Dictionary with monitoring stats
    """
    return LiteLLMMonitor.get_stats()


def print_monitoring_summary():
    """Print monitoring summary to console."""
    LiteLLMMonitor.print_summary()

