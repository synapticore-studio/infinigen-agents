# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Constraints for Infinigen

import math
from dataclasses import dataclass
from typing import List, Tuple

import gin
import numpy as np

from infinigen.core.constraints.constraint_language.expression import (
    BoolExpression,
    ScalarExpression,
)
from infinigen.core.constraints.constraint_language.result import Problem
from infinigen.core.constraints.constraint_language.types import Node


@dataclass
class OrbitalConstraints(Node):
    """Constraints for realistic orbital mechanics"""

    parent_mass: float
    moon_mass: float
    orbital_radius: float
    orbital_period: float
    eccentricity: float = 0.0  # 0 = circular, 1 = parabolic

    def kepler_third_law(self) -> float:
        """Calculate orbital period using Kepler's third law"""
        # T^2 = (4π^2 * a^3) / (G * M)
        # Simplified for our purposes
        G = 6.674e-11  # Gravitational constant
        return math.sqrt(
            (4 * math.pi**2 * self.orbital_radius**3) / (G * self.parent_mass)
        )

    def orbital_velocity(self) -> float:
        """Calculate orbital velocity"""
        G = 6.674e-11
        return math.sqrt(G * self.parent_mass / self.orbital_radius)

    def is_stable_orbit(self) -> bool:
        """Check if orbit is stable (not too close to parent)"""
        # Roche limit approximation
        roche_limit = (
            2.456
            * self.parent_mass ** (1 / 3)
            * (self.moon_mass / self.parent_mass) ** (-1 / 3)
        )
        return self.orbital_radius > roche_limit

    def tidal_locking_distance(self) -> float:
        """Calculate distance where tidal locking occurs"""
        # Simplified tidal locking calculation
        return 10 * self.parent_mass ** (1 / 3)


@dataclass
class PlanetarySystemConstraints(Node):
    """Constraints for entire planetary systems"""

    central_star_mass: float
    planets: List[Tuple[float, float]]  # (mass, distance) pairs
    habitable_zone_inner: float
    habitable_zone_outer: float

    def hill_sphere_radius(self, planet_mass: float, planet_distance: float) -> float:
        """Calculate Hill sphere radius for planet"""
        return planet_distance * (planet_mass / (3 * self.central_star_mass)) ** (1 / 3)

    def is_in_habitable_zone(self, distance: float) -> bool:
        """Check if distance is in habitable zone"""
        return self.habitable_zone_inner <= distance <= self.habitable_zone_outer

    def orbital_resonance(self, inner_period: float, outer_period: float) -> float:
        """Calculate orbital resonance ratio"""
        return outer_period / inner_period


@gin.configurable
class AstronomicalConstraintSolver:
    """Solver for astronomical constraints"""

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve_orbital_system(self, constraints: PlanetarySystemConstraints) -> Problem:
        """Solve constraints for a planetary system"""

        # Create constraint expressions
        constraints_dict = {}
        score_terms = {}

        # Stability constraints
        for i, (mass, distance) in enumerate(constraints.planets):
            # Hill sphere constraint
            hill_radius = constraints.hill_sphere_radius(mass, distance)
            constraints_dict[f"hill_sphere_{i}"] = BoolExpression(
                f"distance_{i} > {hill_radius}"
            )

            # Habitable zone constraint
            if constraints.is_in_habitable_zone(distance):
                constraints_dict[f"habitable_zone_{i}"] = BoolExpression("True")

            # Orbital resonance scoring
            if i > 0:
                inner_period = self._calculate_period(
                    constraints.central_star_mass, constraints.planets[i - 1][1]
                )
                outer_period = self._calculate_period(
                    constraints.central_star_mass, distance
                )
                resonance = constraints.orbital_resonance(inner_period, outer_period)

                # Prefer simple resonances (2:1, 3:2, etc.)
                simple_resonance_score = 1.0 / (1.0 + abs(resonance - round(resonance)))
                score_terms[f"resonance_{i}"] = ScalarExpression(
                    str(simple_resonance_score)
                )

        return Problem(constraints_dict, score_terms)

    def _calculate_period(self, central_mass: float, distance: float) -> float:
        """Calculate orbital period"""
        G = 6.674e-11
        return math.sqrt((4 * math.pi**2 * distance**3) / (G * central_mass))

    def optimize_moon_orbits(
        self, parent_planet_mass: float, moons: List[Tuple[float, float]]
    ) -> List[float]:
        """Optimize moon orbital distances for stability"""
        optimized_distances = []

        for moon_mass, initial_distance in moons:
            # Start with initial distance
            distance = initial_distance

            # Apply constraints
            orbital_constraints = OrbitalConstraints(
                parent_mass=parent_planet_mass,
                moon_mass=moon_mass,
                orbital_radius=distance,
            )

            # Ensure stability
            if not orbital_constraints.is_stable_orbit():
                # Move to minimum stable distance
                distance = orbital_constraints.tidal_locking_distance()

            optimized_distances.append(distance)

        return optimized_distances


@gin.configurable
class RealisticOrbitalMechanics:
    """Realistic orbital mechanics calculations"""

    def __init__(self, gravitational_constant: float = 6.674e-11):
        self.G = gravitational_constant

    def calculate_orbital_elements(
        self,
        central_mass: float,
        satellite_mass: float,
        distance: float,
        velocity: float = None,
    ) -> dict:
        """Calculate orbital elements from basic parameters"""

        if velocity is None:
            # Circular orbit velocity
            velocity = math.sqrt(self.G * central_mass / distance)

        # Semi-major axis
        a = distance

        # Orbital period
        T = 2 * math.pi * math.sqrt(a**3 / (self.G * central_mass))

        # Angular velocity
        omega = 2 * math.pi / T

        # Specific angular momentum
        h = distance * velocity

        # Eccentricity (simplified for circular orbits)
        e = 0.0

        return {
            "semi_major_axis": a,
            "eccentricity": e,
            "period": T,
            "angular_velocity": omega,
            "velocity": velocity,
            "specific_angular_momentum": h,
        }

    def calculate_tidal_forces(
        self, primary_mass: float, secondary_mass: float, distance: float
    ) -> dict:
        """Calculate tidal forces between two bodies"""

        # Tidal acceleration
        tidal_acceleration = 2 * self.G * secondary_mass * distance / (distance**3)

        # Tidal locking time (simplified)
        # This is a very rough approximation
        tidal_locking_time = 1e9 * (distance**6) / (primary_mass * secondary_mass)

        return {
            "tidal_acceleration": tidal_acceleration,
            "tidal_locking_time": tidal_locking_time,
            "is_tidally_locked": tidal_locking_time < 1e6,  # 1 million years
        }

    def calculate_roche_limit(
        self, primary_mass: float, secondary_mass: float, secondary_density: float
    ) -> float:
        """Calculate Roche limit for tidal disruption"""

        # Roche limit formula
        roche_limit = 2.456 * (primary_mass / secondary_density) ** (1 / 3)

        return roche_limit
        # Roche limit formula
        roche_limit = 2.456 * (primary_mass / secondary_density) ** (1 / 3)

        return roche_limit
