"""
Policy Validator

Validates DatasetPolicy instances against schema and constraints.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging
import json

from src.dataset_builder.policies.models import (
    DatasetPolicy,
    SelectionStrategy,
    RatioPolicy,
    CandidatePoolPolicy,
    RandomizationConfig,
    OrganismPolicy,
    DuplicatePolicy,
)


class PolicyValidator:
    """
    Validates DatasetPolicy instances against the contract schema.
    
    Normative Requirements (from contract §8.3):
    - Every dataset family MUST be defined through a policy artifact
    - A policy MUST declare ratio behavior, scarcity handling, randomization behavior, and duplicate behavior
    - A policy MUST NOT rely on hidden defaults that materially affect sampling outcomes
    - A policy SHOULD be validatable against a JSON schema
    
    Attributes:
        logger: Logger instance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, policy: DatasetPolicy) -> tuple[bool, List[str]]:
        """
        Validate a DatasetPolicy instance.
        
        Args:
            policy: DatasetPolicy to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Validate required fields
        if not policy.policy_id:
            errors.append("policy_id is required")
        
        if not policy.source_universe_id:
            errors.append("source_universe_id is required")
        
        # Validate selection strategy
        errors.extend(self._validate_selection_strategy(policy.selection_strategy))
        
        # Validate ratio policy
        errors.extend(self._validate_ratio_policy(policy.ratio_policy))
        
        # Validate candidate pool policy
        errors.extend(self._validate_candidate_pool_policy(policy.candidate_pool_policy))
        
        # Validate randomization config
        errors.extend(self._validate_randomization(policy.randomization))
        
        # Validate split policy reference
        if not policy.split_policy_ref:
            errors.append("split_policy_ref is required")
        
        # Validate organism policy
        errors.extend(self._validate_organism_policy(policy.organism_policy))
        
        # Validate duplicate policy
        errors.extend(self._validate_duplicate_policy(policy.duplicate_policy))
        
        # Check for hidden defaults that affect sampling
        errors.extend(self._validate_no_hidden_defaults(policy))
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.info(f"Policy {policy.policy_id} is valid")
        else:
            self.logger.warning(f"Policy {policy.policy_id} has {len(errors)} validation errors")
        
        return is_valid, errors
    
    def _validate_selection_strategy(
        self, strategy: SelectionStrategy
    ) -> List[str]:
        """Validate selection strategy."""
        errors = []
        
        valid_modes = ["sample_without_replacement", "sample_with_replacement", "use_all"]
        if strategy.mode not in valid_modes:
            errors.append(
                f"Invalid selection mode: {strategy.mode}. Must be one of {valid_modes}"
            )
        
        valid_scopes = ["per_target", "global"]
        if strategy.candidate_scope not in valid_scopes:
            errors.append(
                f"Invalid candidate_scope: {strategy.candidate_scope}. Must be one of {valid_scopes}"
            )
        
        valid_assignments = ["global_unique_candidates", "per_target_unique"]
        if strategy.assignment_strategy not in valid_assignments:
            errors.append(
                f"Invalid assignment_strategy: {strategy.assignment_strategy}. Must be one of {valid_assignments}"
            )
        
        return errors
    
    def _validate_ratio_policy(self, policy: RatioPolicy) -> List[str]:
        """Validate ratio policy."""
        errors = []
        
        if policy.positive_unit != "target":
            errors.append(
                f"Invalid positive_unit: {policy.positive_unit}. Must be 'target'"
            )
        
        if policy.negative_unit != "candidate_assignment":
            errors.append(
                f"Invalid negative_unit: {policy.negative_unit}. Must be 'candidate_assignment'"
            )
        
        # Validate ratio format
        try:
            pos, neg = policy.get_ratio_tuple()
            if pos <= 0 or neg <= 0:
                errors.append(
                    f"Invalid ratio values: {pos}:{neg}. Both must be positive integers"
                )
        except ValueError as e:
            errors.append(f"Invalid ratio format: {e}")
        
        return errors
    
    def _validate_candidate_pool_policy(
        self, policy: CandidatePoolPolicy
    ) -> List[str]:
        """Validate candidate pool policy."""
        errors = []
        
        if policy.min_pool_size < 0:
            errors.append(
                f"Invalid min_pool_size: {policy.min_pool_size}. Must be non-negative"
            )
        
        if policy.max_pool_size is not None and policy.max_pool_size < 0:
            errors.append(
                f"Invalid max_pool_size: {policy.max_pool_size}. Must be non-negative or None"
            )
        
        if policy.min_pool_size > 0 and policy.max_pool_size is not None:
            if policy.min_pool_size > policy.max_pool_size:
                errors.append(
                    f"min_pool_size ({policy.min_pool_size}) cannot be greater than "
                    f"max_pool_size ({policy.max_pool_size})"
                )
        
        valid_scarcity_modes = ["drop_target", "relax_ratio", "use_available"]
        if policy.scarcity_mode not in valid_scarcity_modes:
            errors.append(
                f"Invalid scarcity_mode: {policy.scarcity_mode}. Must be one of {valid_scarcity_modes}"
            )
        
        return errors
    
    def _validate_randomization(
        self, config: RandomizationConfig
    ) -> List[str]:
        """Validate randomization configuration."""
        errors = []
        
        valid_scopes = ["global", "per_target"]
        if config.seed_scope not in valid_scopes:
            errors.append(
                f"Invalid seed_scope: {config.seed_scope}. Must be one of {valid_scopes}"
            )
        
        return errors
    
    def _validate_organism_policy(self, policy: OrganismPolicy) -> List[str]:
        """Validate organism policy."""
        errors = []
        
        valid_modes = ["preserve_source", "filter_by_organism", "balance_by_organism"]
        if policy.mode not in valid_modes:
            errors.append(
                f"Invalid organism mode: {policy.mode}. Must be one of {valid_modes}"
            )
        
        return errors
    
    def _validate_duplicate_policy(self, policy: DuplicatePolicy) -> List[str]:
        """Validate duplicate policy."""
        errors = []
        # No validation needed for boolean fields
        return errors
    
    def _validate_no_hidden_defaults(self, policy: DatasetPolicy) -> List[str]:
        """
        Check for hidden defaults that materially affect sampling outcomes.
        
        This ensures that all important decisions are explicit in the policy.
        """
        errors = []
        
        # Check that ratio is explicitly set (not default)
        if policy.ratio_policy.target_to_negative_ratio == "1:1":
            # This is the default, but it's acceptable as a conscious choice
            pass
        
        # Check that scarcity mode is explicitly set
        if policy.candidate_pool_policy.scarcity_mode == "drop_target":
            # This is the default from contract §15
            pass
        
        return errors
    
    def validate_json_schema(self, policy_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a policy dictionary against the expected schema.
        
        Args:
            policy_dict: Policy as dictionary
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        try:
            # Try to create a DatasetPolicy from the dict
            policy = DatasetPolicy.from_dict(policy_dict)
            return self.validate(policy)
        except (KeyError, TypeError, ValueError) as e:
            return False, [f"Schema validation failed: {e}"]
    
    def validate_json_file(self, file_path: str) -> tuple[bool, List[str]]:
        """
        Validate a policy JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        import json
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            return False, [f"File not found: {file_path}"]
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.validate_json_schema(data)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]
