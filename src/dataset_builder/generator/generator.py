"""
Dataset Variant Generator

Generates concrete dataset variants from universes and policies.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import field
import logging
from datetime import datetime

from src.dataset_builder.input.models import UniverseRecord
from src.dataset_builder.policies.models import DatasetPolicy
from src.dataset_builder.generator.models import (
    DatasetVariant,
    AssignmentRecord,
    VariantManifest,
    ScarcityEvent,
)


class DatasetVariantGenerator:
    """
    Generates concrete dataset variants from a universe and policy.
    
    Normative Requirements (from contract §9.3):
    - One variant MUST be reconstructible from source_universe + policy + seed
    - Variant generation MUST produce both realized instances and a machine-readable variant manifest
    - The generator MUST record dropped targets and scarcity events
    - The generator MUST NOT silently relax ratio rules in v0.1
    - The generator MUST forbid candidate reuse within the same variant
    - The generator SHOULD expose deterministic replay for any variant_id
    
    Attributes:
        logger: Logger instance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate(
        self,
        universe: List[UniverseRecord],
        policy: DatasetPolicy,
        seed: int,
        variant_id: Optional[str] = None,
    ) -> DatasetVariant:
        """
        Generate a dataset variant from a universe and policy.
        
        Args:
            universe: List of UniverseRecord instances
            policy: DatasetPolicy to apply
            seed: Random seed for reproducibility
            variant_id: Optional variant ID (generated if None)
            
        Returns:
            DatasetVariant instance
        """
        # Set random seed for reproducibility
        rng = random.Random(seed)
        
        # Generate variant_id if not provided
        if variant_id is None:
            variant_id = f"variant_{seed:06d}"
        
        # Parse ratio
        pos_count, neg_count = policy.ratio_policy.get_ratio_tuple()
        
        # Initialize tracking
        assignments: List[AssignmentRecord] = []
        scarcity_events: List[Dict[str, Any]] = []
        targets_included = 0
        targets_dropped = 0
        used_candidates: Set[str] = set()  # Track used candidates to prevent reuse
        
        # Process each target
        for record in universe:
            target_id = record.target_id
            candidate_ids = record.candidate_ids
            
            # Check minimum pool size
            min_pool = policy.candidate_pool_policy.min_pool_size
            if len(candidate_ids) < min_pool:
                # Scarcity handling
                if policy.candidate_pool_policy.scarcity_mode == "drop_target":
                    scarcity_events.append({
                        "target_id": target_id,
                        "reason": "insufficient_candidates",
                        "details": f"Need {min_pool} candidates, have {len(candidate_ids)}",
                    })
                    targets_dropped += 1
                    continue
                elif policy.candidate_pool_policy.scarcity_mode == "relax_ratio":
                    # Use available candidates (not implemented in v0.1 per contract)
                    self.logger.warning(
                        f"relax_ratio mode not implemented in v0.1 for target {target_id}"
                    )
                    continue
                else:
                    # use_available
                    pass
            
            # Add positive instance (one per target)
            positive_assignment = AssignmentRecord(
                target_id=target_id,
                protein_id=target_id,  # Positive is the target itself
                role="positive",
                paired_target_id=target_id,
                variant_id=variant_id,
            )
            assignments.append(positive_assignment)
            targets_included += 1
            
            # Sample negative instances
            available_candidates = [
                cid for cid in candidate_ids
                if cid not in used_candidates
            ]
            
            # Check if we have enough candidates
            if len(available_candidates) < neg_count:
                if policy.candidate_pool_policy.scarcity_mode == "drop_target":
                    # Remove the positive we just added
                    assignments = [a for a in assignments if a.target_id != target_id]
                    targets_included -= 1
                    scarcity_events.append({
                        "target_id": target_id,
                        "reason": "insufficient_candidates_for_ratio",
                        "details": f"Need {neg_count} unique candidates, have {len(available_candidates)}",
                    })
                    targets_dropped += 1
                    continue
            
            # Sample candidates without replacement
            if policy.selection_strategy.mode == "sample_without_replacement":
                sampled_candidates = rng.sample(available_candidates, neg_count)
            elif policy.selection_strategy.mode == "sample_with_replacement":
                sampled_candidates = [
                    rng.choice(available_candidates) for _ in range(neg_count)
                ]
            else:  # use_all
                sampled_candidates = available_candidates[:neg_count]
            
            # Add negative assignments
            for candidate_id in sampled_candidates:
                negative_assignment = AssignmentRecord(
                    target_id=target_id,
                    protein_id=candidate_id,
                    role="negative",
                    paired_target_id=target_id,
                    variant_id=variant_id,
                )
                assignments.append(negative_assignment)
                used_candidates.add(candidate_id)
        
        # Calculate statistics
        total_positive = len([a for a in assignments if a.role == "positive"])
        total_negative = len([a for a in assignments if a.role == "negative"])
        
        # Calculate realized ratio
        if total_positive > 0:
            ratio_realized = f"{total_positive}:{total_negative // total_positive}"
        else:
            ratio_realized = "0:0"
        
        # Build manifest
        manifest = VariantManifest(
            variant_id=variant_id,
            policy_id=policy.policy_id,
            source_universe_id=policy.source_universe_id,
            seed_used=seed,
            targets_included=targets_included,
            targets_dropped=targets_dropped,
            total_positive_instances=total_positive,
            total_negative_instances=total_negative,
            assignment_mode=policy.selection_strategy.assignment_strategy,
            scarcity_events=scarcity_events,
            dataset_statistics={
                "ratio_realized": ratio_realized,
                "organism_distribution": self._calculate_organism_distribution(universe, assignments),
                "candidate_pool_size_distribution": self._calculate_pool_size_distribution(universe),
            },
        )
        
        self.logger.info(
            f"Generated variant {variant_id}: "
            f"{targets_included} targets, {total_positive} positive, {total_negative} negative, "
            f"{targets_dropped} dropped"
        )
        
        return DatasetVariant(
            variant_id=variant_id,
            policy_id=policy.policy_id,
            source_universe_id=policy.source_universe_id,
            seed=seed,
            assignments=assignments,
            manifest=manifest,
        )
    
    def generate_multiple(
        self,
        universe: List[UniverseRecord],
        policy: DatasetPolicy,
        num_variants: Optional[int] = None,
    ) -> List[DatasetVariant]:
        """
        Generate multiple variants from the same universe and policy.
        
        Args:
            universe: List of UniverseRecord instances
            policy: DatasetPolicy to apply
            num_variants: Number of variants to generate (default: from policy)
            
        Returns:
            List of DatasetVariant instances
        """
        if num_variants is None:
            num_variants = policy.get_variant_count()
        
        variants = []
        for i in range(num_variants):
            seed = hash((policy.policy_id, i)) % (2**32)
            variant = self.generate(
                universe=universe,
                policy=policy,
                seed=seed,
                variant_id=f"{policy.policy_id}_variant_{i:04d}",
            )
            variants.append(variant)
        
        self.logger.info(f"Generated {len(variants)} variants for policy {policy.policy_id}")
        return variants
    
    def _calculate_organism_distribution(
        self,
        universe: List[UniverseRecord],
        assignments: List[AssignmentRecord],
    ) -> Dict[str, int]:
        """Calculate organism distribution from assignments."""
        # Map target_id to organism
        target_organism = {}
        for record in universe:
            if record.organism:
                target_organism[record.target_id] = record.organism
        
        # Count organisms in assignments
        dist: Dict[str, int] = {}
        for assignment in assignments:
            org = target_organism.get(assignment.target_id)
            if org:
                dist[org] = dist.get(org, 0) + 1
        
        return dist
    
    def _calculate_pool_size_distribution(
        self,
        universe: List[UniverseRecord],
    ) -> Dict[int, int]:
        """Calculate candidate pool size distribution."""
        dist: Dict[int, int] = {}
        for record in universe:
            size = record.candidate_count
            dist[size] = dist.get(size, 0) + 1
        return dist
    
    def generate_for_ratio_families(
        self,
        universe: List[UniverseRecord],
        base_policy: DatasetPolicy,
        ratio_families: Optional[List[str]] = None,
    ) -> List[DatasetVariant]:
        """
        Generate variants for multiple ratio families.
        
        Args:
            universe: List of UniverseRecord instances
            base_policy: Base policy (ratio_policy will be overridden)
            ratio_families: List of ratio strings (default: from contract §15)
            
        Returns:
            List of DatasetVariant instances (one per ratio family)
        """
        if ratio_families is None:
            ratio_families = base_policy.get_ratio_families()
        
        variants = []
        for ratio in ratio_families:
            # Create policy with this ratio
            policy = DatasetPolicy(
                policy_id=f"{base_policy.policy_id}_ratio_{ratio.replace(':', 'to')}",
                source_universe_id=base_policy.source_universe_id,
                selection_strategy=base_policy.selection_strategy,
                ratio_policy=base_policy.ratio_policy.__class__(
                    positive_unit=base_policy.ratio_policy.positive_unit,
                    negative_unit=base_policy.ratio_policy.negative_unit,
                    target_to_negative_ratio=ratio,
                ),
                candidate_pool_policy=base_policy.candidate_pool_policy,
                randomization=base_policy.randomization,
                split_policy_ref=base_policy.split_policy_ref,
                organism_policy=base_policy.organism_policy,
                duplicate_policy=base_policy.duplicate_policy,
            )
            
            # Generate one variant for this ratio
            variant = self.generate(
                universe=universe,
                policy=policy,
                seed=42,  # Fixed seed for reproducibility
                variant_id=f"{policy.policy_id}_seed42",
            )
            variants.append(variant)
        
        self.logger.info(f"Generated variants for {len(ratio_families)} ratio families")
        return variants
