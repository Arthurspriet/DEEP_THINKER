"""
YAML Loader for Bio Pattern Cards.

Loads and validates BioPattern YAML files from the patterns directory.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .schema import BioPattern, BioPatternValidationError

logger = logging.getLogger(__name__)


# Default patterns directory (relative to this module)
PATTERNS_DIR = Path(__file__).parent / "patterns"


def load_patterns(
    patterns_dir: Optional[Path] = None,
    validate: bool = True,
) -> List[BioPattern]:
    """
    Load all BioPattern YAML files from directory.
    
    Args:
        patterns_dir: Directory containing YAML files (default: module patterns/)
        validate: Whether to validate patterns (default: True)
        
    Returns:
        List of BioPattern instances
        
    Raises:
        BioPatternValidationError: If validation is enabled and fails
    """
    if patterns_dir is None:
        patterns_dir = PATTERNS_DIR
    
    patterns: List[BioPattern] = []
    
    if not patterns_dir.exists():
        logger.warning(f"[BIO_PRIORS] Patterns directory not found: {patterns_dir}")
        return patterns
    
    yaml_files = sorted(patterns_dir.glob("*.yaml"))
    
    if not yaml_files:
        logger.warning(f"[BIO_PRIORS] No YAML files found in {patterns_dir}")
        return patterns
    
    for yaml_file in yaml_files:
        try:
            pattern = load_pattern_file(yaml_file, validate=validate)
            patterns.append(pattern)
            logger.debug(f"[BIO_PRIORS] Loaded pattern: {pattern.id}")
        except BioPatternValidationError as e:
            logger.error(f"[BIO_PRIORS] Validation error in {yaml_file}: {e}")
            if validate:
                raise
        except Exception as e:
            logger.error(f"[BIO_PRIORS] Error loading {yaml_file}: {e}")
            if validate:
                raise
    
    logger.info(f"[BIO_PRIORS] Loaded {len(patterns)} patterns from {patterns_dir}")
    return patterns


def load_pattern_file(
    yaml_file: Path,
    validate: bool = True,
) -> BioPattern:
    """
    Load a single BioPattern from a YAML file.
    
    Args:
        yaml_file: Path to YAML file
        validate: Whether to validate pattern
        
    Returns:
        BioPattern instance
        
    Raises:
        BioPatternValidationError: If validation fails
    """
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise BioPatternValidationError(f"Empty YAML file: {yaml_file}")
    
    # Handle both flat and nested formats
    if "pattern" in data:
        data = data["pattern"]
    
    pattern = BioPattern.from_dict(data)
    
    return pattern


def validate_pattern(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate pattern data without creating BioPattern.
    
    Args:
        data: Pattern data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: List[str] = []
    
    # Required fields
    required = ["id", "name", "problem_class", "conditions", "mechanism", "system_mapping"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # ID validation
    if not data["id"].startswith("BIO_"):
        errors.append(f"id must start with 'BIO_', got '{data['id']}'")
    
    # Weight validation
    weight = data.get("weight", 0.3)
    if not 0.0 <= weight <= 1.0:
        errors.append(f"weight must be in [0, 1], got {weight}")
    
    # Maturity validation
    maturity = data.get("maturity", "draft")
    if maturity not in {"draft", "stable"}:
        errors.append(f"maturity must be 'draft' or 'stable', got '{maturity}'")
    
    # problem_class validation
    if not data["problem_class"]:
        errors.append("must have at least 1 problem_class")
    
    # system_mapping validation
    if not data["system_mapping"]:
        errors.append("must have at least 1 system_mapping key")
    else:
        from .schema import VALID_SYSTEM_MAPPING_KEYS
        invalid_keys = set(data["system_mapping"].keys()) - VALID_SYSTEM_MAPPING_KEYS
        if invalid_keys:
            errors.append(f"system_mapping has invalid keys: {invalid_keys}")
    
    return len(errors) == 0, errors


def validate_all_patterns(
    patterns_dir: Optional[Path] = None,
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate all YAML files in patterns directory.
    
    Args:
        patterns_dir: Directory containing YAML files
        
    Returns:
        Tuple of (all_valid, errors_by_file)
    """
    if patterns_dir is None:
        patterns_dir = PATTERNS_DIR
    
    errors_by_file: Dict[str, List[str]] = {}
    
    if not patterns_dir.exists():
        return False, {"_directory": [f"Directory not found: {patterns_dir}"]}
    
    yaml_files = sorted(patterns_dir.glob("*.yaml"))
    
    if not yaml_files:
        return False, {"_directory": [f"No YAML files found in {patterns_dir}"]}
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if data is None:
                errors_by_file[str(yaml_file)] = ["Empty YAML file"]
                continue
            
            if "pattern" in data:
                data = data["pattern"]
            
            is_valid, errors = validate_pattern(data)
            if not is_valid:
                errors_by_file[str(yaml_file)] = errors
                
        except yaml.YAMLError as e:
            errors_by_file[str(yaml_file)] = [f"YAML parse error: {e}"]
        except Exception as e:
            errors_by_file[str(yaml_file)] = [f"Error: {e}"]
    
    return len(errors_by_file) == 0, errors_by_file


def get_patterns_summary(
    patterns_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Get summary of all patterns in directory.
    
    Args:
        patterns_dir: Directory containing YAML files
        
    Returns:
        List of pattern summaries (id, name, maturity, weight)
    """
    patterns = load_patterns(patterns_dir, validate=False)
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "maturity": p.maturity,
            "weight": p.weight,
            "problem_classes": p.problem_class,
        }
        for p in patterns
    ]

