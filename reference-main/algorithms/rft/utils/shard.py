#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# MIGRATED FROM: algorithms/rft/core/shard.py
# IMPORT UPDATE: .geometric_container and .bloom_filter now within utils/
"""
Shard — quantoniumos RFT utils

Shard structure for grouping GeometricContainers with Bloom filter pre-check.
"""

from typing import List
from .geometric_container import GeometricContainer
from .bloom_filter import SimplifiedBloomFilter, hash1, hash2


class Shard:
    def __init__(self, containers: List[GeometricContainer]):
        self.containers = containers
        self.bloom = self._build_bloom(containers)

    def _build_bloom(self, containers: List[GeometricContainer]) -> SimplifiedBloomFilter:
        bf = SimplifiedBloomFilter(256, [hash1, hash2])
        for c in containers:
            if c.resonant_frequencies:
                bf.add(c.resonant_frequencies[0])
        return bf

    def search(self, target_freq: float, threshold: float = 0.1) -> List[GeometricContainer]:
        if not self.bloom.test(target_freq):
            return []
        results = []
        for c in self.containers:
            if not c.resonant_frequencies:
                continue
            if abs(c.resonant_frequencies[0] - target_freq) < threshold:
                results.append(c)
        return results
