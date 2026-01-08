"""
Model Cost Database for DeepThinker 2.0.

Contains approximate VRAM requirements, load times, and inference speeds
for common LLM models used with Ollama.

All values are estimates and may vary based on:
- Quantization level (Q4, Q5, Q8, FP16)
- Context length
- Batch size
- GPU architecture
"""

from typing import Dict, Any

# Model cost database
# Format: model_name -> {vram_mb, load_time_s, tokens_per_sec, tier}
MODEL_COSTS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Small Models (< 5GB VRAM) - Fast inference, good for simple tasks
    # =========================================================================
    "llama3.2:1b": {
        "vram_mb": 1500,
        "load_time_s": 3,
        "tokens_per_sec": 150,
        "tier": "small",
        "description": "Very fast, lightweight model for simple tasks"
    },
    "llama3.2:3b": {
        "vram_mb": 2500,
        "load_time_s": 4,
        "tokens_per_sec": 120,
        "tier": "small",
        "description": "Fast model with decent reasoning"
    },
    "phi3:mini": {
        "vram_mb": 2800,
        "load_time_s": 4,
        "tokens_per_sec": 110,
        "tier": "small",
        "description": "Microsoft's efficient small model"
    },
    "phi3:3.8b": {
        "vram_mb": 3000,
        "load_time_s": 5,
        "tokens_per_sec": 100,
        "tier": "small",
        "description": "Phi-3 medium variant"
    },
    "gemma2:2b": {
        "vram_mb": 2000,
        "load_time_s": 3,
        "tokens_per_sec": 130,
        "tier": "small",
        "description": "Google's efficient small model"
    },
    "qwen2.5:0.5b": {
        "vram_mb": 800,
        "load_time_s": 2,
        "tokens_per_sec": 200,
        "tier": "small",
        "description": "Extremely lightweight for basic tasks"
    },
    "qwen2.5:1.5b": {
        "vram_mb": 1500,
        "load_time_s": 3,
        "tokens_per_sec": 140,
        "tier": "small",
        "description": "Small but capable Qwen variant"
    },
    "qwen2.5:3b": {
        "vram_mb": 2500,
        "load_time_s": 4,
        "tokens_per_sec": 110,
        "tier": "small",
        "description": "Balanced small Qwen model"
    },
    
    # =========================================================================
    # Medium Models (5-12GB VRAM) - Balanced performance
    # =========================================================================
    "llama3.1:8b": {
        "vram_mb": 6000,
        "load_time_s": 8,
        "tokens_per_sec": 70,
        "tier": "medium",
        "description": "Strong general-purpose model"
    },
    "llama3.2:8b": {
        "vram_mb": 6000,
        "load_time_s": 8,
        "tokens_per_sec": 70,
        "tier": "medium",
        "description": "Latest Llama 8B variant"
    },
    "deepseek-r1:8b": {
        "vram_mb": 6500,
        "load_time_s": 10,
        "tokens_per_sec": 55,
        "tier": "medium",
        "description": "Excellent for code and reasoning"
    },
    "deepseek-coder:6.7b": {
        "vram_mb": 5500,
        "load_time_s": 8,
        "tokens_per_sec": 65,
        "tier": "medium",
        "description": "Specialized coding model"
    },
    "qwen2.5:7b": {
        "vram_mb": 5500,
        "load_time_s": 7,
        "tokens_per_sec": 75,
        "tier": "medium",
        "description": "Strong Qwen variant"
    },
    "qwen2.5-coder:7b": {
        "vram_mb": 5500,
        "load_time_s": 7,
        "tokens_per_sec": 70,
        "tier": "medium",
        "description": "Qwen coding specialist"
    },
    "gemma2:9b": {
        "vram_mb": 7000,
        "load_time_s": 10,
        "tokens_per_sec": 60,
        "tier": "medium",
        "description": "Google's capable 9B model"
    },
    "mistral:7b": {
        "vram_mb": 5500,
        "load_time_s": 7,
        "tokens_per_sec": 75,
        "tier": "medium",
        "description": "Efficient Mistral base model"
    },
    "mistral:instruct": {
        "vram_mb": 5500,
        "load_time_s": 7,
        "tokens_per_sec": 75,
        "tier": "medium",
        "description": "Mistral instruction-tuned"
    },
    "codellama:7b": {
        "vram_mb": 5500,
        "load_time_s": 7,
        "tokens_per_sec": 70,
        "tier": "medium",
        "description": "Meta's code-specialized Llama"
    },
    "cogito:8b": {
        "vram_mb": 6000,
        "load_time_s": 8,
        "tokens_per_sec": 65,
        "tier": "medium",
        "description": "Reasoning-focused model"
    },
    
    # =========================================================================
    # Large Models (12-25GB VRAM) - High capability
    # =========================================================================
    "gemma3:12b": {
        "vram_mb": 9000,
        "load_time_s": 12,
        "tokens_per_sec": 50,
        "tier": "large",
        "description": "Google's 12B Gemma"
    },
    "llama3.1:13b": {
        "vram_mb": 10000,
        "load_time_s": 15,
        "tokens_per_sec": 45,
        "tier": "large",
        "description": "Llama 13B variant"
    },
    "cogito:14b": {
        "vram_mb": 11000,
        "load_time_s": 18,
        "tokens_per_sec": 40,
        "tier": "large",
        "description": "Strong reasoning at 14B"
    },
    "qwen2.5:14b": {
        "vram_mb": 11000,
        "load_time_s": 16,
        "tokens_per_sec": 42,
        "tier": "large",
        "description": "Qwen 14B variant"
    },
    "deepseek-coder:33b": {
        "vram_mb": 22000,
        "load_time_s": 35,
        "tokens_per_sec": 28,
        "tier": "large",
        "description": "Large coding specialist"
    },
    "gemma3:27b": {
        "vram_mb": 18000,
        "load_time_s": 25,
        "tokens_per_sec": 32,
        "tier": "large",
        "description": "Google's large Gemma model"
    },
    "mixtral:8x7b": {
        "vram_mb": 26000,
        "load_time_s": 40,
        "tokens_per_sec": 35,
        "tier": "large",
        "description": "Mixture of experts model"
    },
    "codellama:34b": {
        "vram_mb": 22000,
        "load_time_s": 35,
        "tokens_per_sec": 25,
        "tier": "large",
        "description": "Large code-specialized model"
    },
    
    # =========================================================================
    # XLarge Models (25GB+ VRAM) - Maximum capability
    # =========================================================================
    "llama3.1:70b": {
        "vram_mb": 42000,
        "load_time_s": 60,
        "tokens_per_sec": 15,
        "tier": "xlarge",
        "description": "Llama's largest model"
    },
    "llama3:70b": {
        "vram_mb": 42000,
        "load_time_s": 60,
        "tokens_per_sec": 15,
        "tier": "xlarge",
        "description": "Llama 3 70B"
    },
    "qwen2.5:72b": {
        "vram_mb": 45000,
        "load_time_s": 70,
        "tokens_per_sec": 12,
        "tier": "xlarge",
        "description": "Qwen's largest model"
    },
    "deepseek-r1:70b": {
        "vram_mb": 44000,
        "load_time_s": 65,
        "tokens_per_sec": 14,
        "tier": "xlarge",
        "description": "DeepSeek's large reasoning model"
    },
    "codellama:70b": {
        "vram_mb": 42000,
        "load_time_s": 60,
        "tokens_per_sec": 14,
        "tier": "xlarge",
        "description": "Largest code Llama"
    },
    "mixtral:8x22b": {
        "vram_mb": 85000,
        "load_time_s": 120,
        "tokens_per_sec": 10,
        "tier": "xlarge",
        "description": "Large mixture of experts"
    },
}


# Model tier recommendations by task type
TIER_RECOMMENDATIONS = {
    "research": {
        "default": "medium",
        "low_pressure": "large",
        "high_pressure": "small",
        "preferred_models": ["gemma3:12b", "qwen2.5:7b", "llama3.1:8b"]
    },
    "planning": {
        "default": "medium",
        "low_pressure": "large",
        "high_pressure": "small",
        "preferred_models": ["cogito:14b", "gemma3:27b", "llama3.1:8b"]
    },
    "coding": {
        "default": "medium",
        "low_pressure": "large",
        "high_pressure": "medium",  # Coding needs at least medium
        "preferred_models": ["deepseek-r1:8b", "qwen2.5-coder:7b", "deepseek-coder:33b"]
    },
    "evaluation": {
        "default": "large",
        "low_pressure": "xlarge",
        "high_pressure": "medium",
        "preferred_models": ["gemma3:27b", "llama3.1:70b", "cogito:14b"]
    },
    "simulation": {
        "default": "medium",
        "low_pressure": "large",
        "high_pressure": "small",
        "preferred_models": ["mistral:instruct", "gemma2:9b", "llama3.1:8b"]
    },
    "synthesis": {
        "default": "large",
        "low_pressure": "xlarge",
        "high_pressure": "medium",
        "preferred_models": ["llama3.1:70b", "gemma3:27b", "qwen2.5:14b"]
    }
}


def get_models_by_tier(tier: str) -> list:
    """
    Get list of models in a specific tier.
    
    Args:
        tier: Model tier ("small", "medium", "large", "xlarge")
        
    Returns:
        List of model names in that tier
    """
    return [
        name for name, cost in MODEL_COSTS.items()
        if cost.get("tier") == tier
    ]


def get_models_under_vram(max_vram_mb: int) -> list:
    """
    Get list of models that fit within VRAM limit.
    
    Args:
        max_vram_mb: Maximum VRAM in MB
        
    Returns:
        List of model names sorted by capability (largest first)
    """
    eligible = [
        (name, cost) for name, cost in MODEL_COSTS.items()
        if cost.get("vram_mb", 0) <= max_vram_mb
    ]
    # Sort by VRAM (proxy for capability) descending
    eligible.sort(key=lambda x: x[1].get("vram_mb", 0), reverse=True)
    return [name for name, _ in eligible]


def get_recommended_models(
    task_type: str,
    available_vram_mb: int,
    pressure: str = "medium"
) -> list:
    """
    Get recommended models for a task given resource constraints.
    
    Args:
        task_type: Type of task (research, planning, coding, etc.)
        available_vram_mb: Available VRAM in MB
        pressure: Resource pressure level
        
    Returns:
        List of recommended model names
    """
    recommendations = TIER_RECOMMENDATIONS.get(task_type, TIER_RECOMMENDATIONS["research"])
    
    if pressure == "high" or pressure == "critical":
        target_tier = recommendations.get("high_pressure", "small")
    elif pressure == "low":
        target_tier = recommendations.get("low_pressure", "large")
    else:
        target_tier = recommendations.get("default", "medium")
    
    # Get models in target tier that fit in VRAM
    tier_models = get_models_by_tier(target_tier)
    fitting_models = [
        m for m in tier_models
        if MODEL_COSTS.get(m, {}).get("vram_mb", 0) <= available_vram_mb
    ]
    
    # Fall back to smaller tier if nothing fits
    tier_order = ["xlarge", "large", "medium", "small"]
    current_idx = tier_order.index(target_tier) if target_tier in tier_order else 2
    
    while not fitting_models and current_idx < len(tier_order) - 1:
        current_idx += 1
        tier_models = get_models_by_tier(tier_order[current_idx])
        fitting_models = [
            m for m in tier_models
            if MODEL_COSTS.get(m, {}).get("vram_mb", 0) <= available_vram_mb
        ]
    
    # Prioritize preferred models
    preferred = recommendations.get("preferred_models", [])
    result = []
    for m in preferred:
        if m in fitting_models:
            result.append(m)
    
    # Add remaining fitting models
    for m in fitting_models:
        if m not in result:
            result.append(m)
    
    return result[:5]  # Return top 5

