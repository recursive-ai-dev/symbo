# Copyright 2025
# Damien Davison & Michael Maillet & Sacha Davison
# Recursive AI Devs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
A* Pathfinding on Symbolic Energy Landscapes
=============================================

This module implements A* pathfinding where nodes are symbolic states
on an energy landscape (manifold), and the heuristic is derived from
variable influence mapping.

The path cost function uses symbolic energy differences between states,
connecting reasoning logic with core Symbo features.

Key Features:
- Symbolic state representation
- Energy-based cost function
- Variable influence heuristics
- Manifold-aware pathfinding
- Integration with symbolic tensors
"""

import sympy as sp
import numpy as np
import heapq
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass(order=True)
class SearchNode:
    """
    A* search node representing a symbolic state.
    
    Attributes
    ----------
    f_score : float
        Total estimated cost (g + h)
    state : Tuple[float, ...]
        State coordinates (not compared in ordering)
    g_score : float
        Cost from start to this node
    h_score : float
        Heuristic estimate to goal
    parent : Optional['SearchNode']
        Parent node in path
    symbolic_state : Dict[sp.Symbol, float]
        Symbolic representation of state
    """
    
    f_score: float
    state: Tuple[float, ...] = field(compare=False)
    g_score: float = field(compare=False)
    h_score: float = field(compare=False)
    parent: Optional['SearchNode'] = field(default=None, compare=False)
    symbolic_state: Dict[sp.Symbol, float] = field(default_factory=dict, compare=False)


class EnergyLandscape(ABC):
    """
    Abstract base class for symbolic energy landscapes.
    
    Defines the interface for computing energies and influences
    on symbolic manifolds.
    """
    
    @abstractmethod
    def energy(self, state: Dict[sp.Symbol, float]) -> float:
        """Compute energy at a symbolic state."""
        pass
    
    @abstractmethod
    def gradient(self, state: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        """Compute gradient of energy."""
        pass
    
    @abstractmethod
    def influence_map(self, state: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        """Compute variable influence at state."""
        pass


class SymbolicEnergyLandscape(EnergyLandscape):
    """
    Energy landscape defined by a symbolic expression.
    
    Parameters
    ----------
    energy_expr : sp.Expr
        Symbolic energy function E(x₁, x₂, ...)
    variables : List[sp.Symbol]
        State variables
    
    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> E = x**2 + y**2 - 2*x*y  # Energy function
    >>> landscape = SymbolicEnergyLandscape(E, [x, y])
    >>> energy = landscape.energy({x: 1.0, y: 0.5})
    """
    
    def __init__(self, energy_expr: sp.Expr, variables: List[sp.Symbol]):
        """Initialize symbolic energy landscape."""
        self.energy_expr = energy_expr
        self.variables = variables
        
        # Precompute gradient expressions
        self._gradient_exprs = {
            var: sp.diff(energy_expr, var) for var in variables
        }
        
        # Precompile for fast evaluation
        self._energy_func = sp.lambdify(variables, energy_expr, modules='numpy')
        self._gradient_funcs = {
            var: sp.lambdify(variables, grad_expr, modules='numpy')
            for var, grad_expr in self._gradient_exprs.items()
        }
    
    def energy(self, state: Dict[sp.Symbol, float]) -> float:
        """Compute energy at state."""
        args = [state.get(var, 0.0) for var in self.variables]
        return float(self._energy_func(*args))
    
    def gradient(self, state: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        """Compute energy gradient."""
        args = [state.get(var, 0.0) for var in self.variables]
        return {
            var: float(func(*args))
            for var, func in self._gradient_funcs.items()
        }
    
    def influence_map(self, state: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        """
        Compute variable influence (gradient magnitude).
        
        Influence measures how much each variable affects the energy
        at the current state.
        """
        grad = self.gradient(state)
        return {var: abs(val) for var, val in grad.items()}


class SymbolicAStarPathfinder:
    """
    A* pathfinding on symbolic energy landscapes.
    
    This class implements A* search where:
    - Nodes are symbolic states on a manifold
    - Edge costs are energy differences
    - Heuristic is derived from variable influence
    
    Parameters
    ----------
    landscape : EnergyLandscape
        Energy landscape to search on
    variables : List[sp.Symbol]
        State variables
    bounds : Dict[sp.Symbol, Tuple[float, float]]
        Variable bounds (min, max)
    step_size : float, optional
        Step size for neighbor generation
    mode : str, optional
        'minimize' to find low-energy paths, 'maximize' for high-energy
        
    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> landscape = SymbolicEnergyLandscape(x**2 + y**2, [x, y])
    >>> pathfinder = SymbolicAStarPathfinder(
    ...     landscape, [x, y],
    ...     bounds={x: (-5, 5), y: (-5, 5)}
    ... )
    >>> path = pathfinder.find_path(
    ...     start={x: -2, y: -2},
    ...     goal={x: 2, y: 2}
    ... )
    """
    
    def __init__(self,
                 landscape: EnergyLandscape,
                 variables: List[sp.Symbol],
                 bounds: Dict[sp.Symbol, Tuple[float, float]],
                 step_size: float = 0.1,
                 mode: str = 'minimize'):
        """Initialize symbolic A* pathfinder."""
        self.landscape = landscape
        self.variables = variables
        self.bounds = bounds
        self.step_size = step_size
        self.mode = mode
        
        # Cost multiplier based on mode
        self._cost_multiplier = 1.0 if mode == 'minimize' else -1.0
    
    def heuristic(self,
                  state: Dict[sp.Symbol, float],
                  goal: Dict[sp.Symbol, float]) -> float:
        """
        Compute heuristic estimate of cost to goal.
        
        Uses a combination of:
        1. Euclidean distance (geometric component)
        2. Energy difference (landscape component)
        3. Influence-weighted distance (gradient component)
        
        Parameters
        ----------
        state : Dict[sp.Symbol, float]
            Current state
        goal : Dict[sp.Symbol, float]
            Goal state
            
        Returns
        -------
        float
            Heuristic cost estimate
        """
        # Geometric distance
        euclidean = np.sqrt(sum(
            (state.get(var, 0) - goal.get(var, 0))**2
            for var in self.variables
        ))
        
        # Energy difference
        try:
            energy_current = self.landscape.energy(state)
            energy_goal = self.landscape.energy(goal)
            energy_diff = abs(energy_goal - energy_current)
        except:
            energy_diff = 0.0
        
        # Influence-weighted component
        try:
            influence = self.landscape.influence_map(state)
            weighted_dist = sum(
                influence.get(var, 1.0) * abs(state.get(var, 0) - goal.get(var, 0))
                for var in self.variables
            )
        except:
            weighted_dist = euclidean
        
        # Combine components
        # Weight geometric distance more to ensure admissibility
        return 0.5 * euclidean + 0.3 * energy_diff + 0.2 * weighted_dist
    
    def edge_cost(self,
                  state1: Dict[sp.Symbol, float],
                  state2: Dict[sp.Symbol, float]) -> float:
        """
        Compute cost of moving from state1 to state2.
        
        Cost is based on:
        1. Energy difference (landscape traversal cost)
        2. Distance (movement cost)
        
        Parameters
        ----------
        state1, state2 : Dict[sp.Symbol, float]
            States
            
        Returns
        -------
        float
            Edge cost
        """
        # Distance component
        distance = np.sqrt(sum(
            (state1.get(var, 0) - state2.get(var, 0))**2
            for var in self.variables
        ))
        
        # Energy component
        try:
            energy1 = self.landscape.energy(state1)
            energy2 = self.landscape.energy(state2)
            
            # Cost depends on mode
            if self.mode == 'minimize':
                # Penalize going uphill
                energy_cost = max(0, energy2 - energy1)
            else:
                # Reward going uphill (negative cost)
                energy_cost = max(0, energy1 - energy2)
        except:
            energy_cost = 0.0
        
        # Combine
        return distance + self._cost_multiplier * energy_cost
    
    def get_neighbors(self, state: Dict[sp.Symbol, float]) -> List[Dict[sp.Symbol, float]]:
        """
        Generate neighboring states.
        
        Creates neighbors by stepping in each variable direction
        within bounds.
        
        Parameters
        ----------
        state : Dict[sp.Symbol, float]
            Current state
            
        Returns
        -------
        List[Dict[sp.Symbol, float]]
            List of neighboring states
        """
        neighbors = []
        
        # Step in positive and negative direction for each variable
        for var in self.variables:
            current_val = state.get(var, 0.0)
            var_min, var_max = self.bounds.get(var, (-np.inf, np.inf))
            
            # Positive step
            new_val_pos = current_val + self.step_size
            if new_val_pos <= var_max:
                neighbor = state.copy()
                neighbor[var] = new_val_pos
                neighbors.append(neighbor)
            
            # Negative step
            new_val_neg = current_val - self.step_size
            if new_val_neg >= var_min:
                neighbor = state.copy()
                neighbor[var] = new_val_neg
                neighbors.append(neighbor)
        
        return neighbors
    
    def state_to_tuple(self, state: Dict[sp.Symbol, float]) -> Tuple[float, ...]:
        """Convert state dict to hashable tuple."""
        return tuple(state.get(var, 0.0) for var in self.variables)
    
    def tuple_to_state(self, state_tuple: Tuple[float, ...]) -> Dict[sp.Symbol, float]:
        """Convert tuple back to state dict."""
        return {var: val for var, val in zip(self.variables, state_tuple)}
    
    def find_path(self,
                  start: Dict[sp.Symbol, float],
                  goal: Dict[sp.Symbol, float],
                  max_iterations: int = 10000,
                  tolerance: float = 0.1) -> List[Dict[sp.Symbol, float]]:
        """
        Find optimal path from start to goal using A*.
        
        Parameters
        ----------
        start : Dict[sp.Symbol, float]
            Starting state
        goal : Dict[sp.Symbol, float]
            Goal state
        max_iterations : int
            Maximum search iterations
        tolerance : float
            Distance tolerance for reaching goal
            
        Returns
        -------
        List[Dict[sp.Symbol, float]]
            Path as list of states from start to goal
            
        Raises
        ------
        ValueError
            If no path found within max_iterations
        """
        # Initialize
        start_tuple = self.state_to_tuple(start)
        goal_tuple = self.state_to_tuple(goal)
        
        # Priority queue: (f_score, counter, node)
        open_set = []
        counter = 0
        
        start_node = SearchNode(
            f_score=self.heuristic(start, goal),
            state=start_tuple,
            g_score=0.0,
            h_score=self.heuristic(start, goal),
            parent=None,
            symbolic_state=start
        )
        
        heapq.heappush(open_set, (start_node.f_score, counter, start_node))
        counter += 1
        
        # Track visited states
        visited: Set[Tuple[float, ...]] = set()
        
        # Best g_score for each state
        g_scores: Dict[Tuple[float, ...], float] = {start_tuple: 0.0}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Check if reached goal
            distance_to_goal = np.sqrt(sum(
                (current.symbolic_state.get(var, 0) - goal.get(var, 0))**2
                for var in self.variables
            ))
            
            if distance_to_goal <= tolerance:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node.symbolic_state)
                    node = node.parent
                path.reverse()
                return path
            
            # Mark as visited
            visited.add(current.state)
            
            # Explore neighbors
            for neighbor_state in self.get_neighbors(current.symbolic_state):
                neighbor_tuple = self.state_to_tuple(neighbor_state)
                
                if neighbor_tuple in visited:
                    continue
                
                # Compute tentative g_score
                edge_cost = self.edge_cost(current.symbolic_state, neighbor_state)
                tentative_g = current.g_score + edge_cost
                
                # Check if this is better path
                if tentative_g < g_scores.get(neighbor_tuple, np.inf):
                    # Create neighbor node
                    h = self.heuristic(neighbor_state, goal)
                    neighbor_node = SearchNode(
                        f_score=tentative_g + h,
                        state=neighbor_tuple,
                        g_score=tentative_g,
                        h_score=h,
                        parent=current,
                        symbolic_state=neighbor_state
                    )
                    
                    g_scores[neighbor_tuple] = tentative_g
                    heapq.heappush(open_set, (neighbor_node.f_score, counter, neighbor_node))
                    counter += 1
        
        raise ValueError(f"No path found within {max_iterations} iterations")
    
    def compute_path_cost(self, path: List[Dict[sp.Symbol, float]]) -> float:
        """Compute total cost of a path."""
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.edge_cost(path[i], path[i + 1])
        return total_cost
    
    def analyze_path(self, path: List[Dict[sp.Symbol, float]]) -> Dict[str, Any]:
        """
        Analyze a path and return detailed information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - length: number of steps
            - cost: total path cost
            - energy_profile: energy at each step
            - distance: total distance traveled
        """
        if not path:
            return {}
        
        energy_profile = [self.landscape.energy(state) for state in path]
        
        total_distance = sum(
            np.sqrt(sum(
                (path[i].get(var, 0) - path[i+1].get(var, 0))**2
                for var in self.variables
            ))
            for i in range(len(path) - 1)
        )
        
        return {
            "length": len(path),
            "cost": self.compute_path_cost(path),
            "energy_profile": energy_profile,
            "distance": total_distance,
            "start_energy": energy_profile[0] if energy_profile else None,
            "end_energy": energy_profile[-1] if energy_profile else None,
            "energy_change": (energy_profile[-1] - energy_profile[0]) if len(energy_profile) > 1 else 0
        }


__all__ = [
    'SearchNode',
    'EnergyLandscape',
    'SymbolicEnergyLandscape',
    'SymbolicAStarPathfinder',
]
