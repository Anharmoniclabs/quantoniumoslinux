#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# MIGRATED FROM: algorithms/rft/core/vibrational_engine.py
# IMPORT UPDATE: .geometric_container now within utils/
"""
Vibrational Engine — quantoniumos RFT utils

Service that checks if a container resonates at a given frequency
and retrieves its encoded data if so.
"""

from .geometric_container import GeometricContainer


class VibrationalEngine:
    def retrieve_data(self, container: GeometricContainer, frequency: float) -> str:
        if container.check_resonance(frequency):
            data = container.get_data()
            return data if data is not None else ''
        return ''
