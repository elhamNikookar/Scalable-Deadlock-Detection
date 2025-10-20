"""
Fuzzy Genetic Algorithm Implementation
Based on Salimi et al. (2020) methodology for reachability verification and deadlock detection
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class FuzzySet:
    """Represents a fuzzy set with triangular membership function"""
    a: float  # Left boundary
    b: float  # Peak
    c: float  # Right boundary
    
    def membership(self, x: float) -> float:
        """Calculate membership degree for value x"""
        if x < self.a or x > self.c:
            return 0.0
        elif self.a <= x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 1.0
        else:  # self.b < x <= self.c
            return (self.c - x) / (self.c - self.b) if self.c != self.b else 1.0


class FuzzyRule:
    """Represents a fuzzy rule"""
    
    def __init__(self, antecedents: List[FuzzySet], consequent: FuzzySet):
        self.antecedents = antecedents
        self.consequent = consequent
        self.firing_strength = 0.0
    
    def evaluate(self, inputs: List[float]) -> float:
        """Evaluate the rule with given inputs"""
        if len(inputs) != len(self.antecedents):
            raise ValueError("Number of inputs must match number of antecedents")
        
        # Calculate firing strength using min operator
        self.firing_strength = min(
            fuzzy_set.membership(input_val) 
            for fuzzy_set, input_val in zip(self.antecedents, inputs)
        )
        
        return self.firing_strength


class Individual:
    """Represents an individual in the genetic algorithm population"""
    
    def __init__(self, chromosome: List[float], fitness: float = 0.0):
        self.chromosome = chromosome
        self.fitness = fitness
        self.age = 0
    
    def __len__(self):
        return len(self.chromosome)
    
    def __getitem__(self, index):
        return self.chromosome[index]
    
    def __setitem__(self, index, value):
        self.chromosome[index] = value


class FuzzyGeneticAlgorithm:
    """
    Fuzzy Genetic Algorithm for reachability verification and deadlock detection
    Based on Salimi et al. (2020) methodology
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 chromosome_length: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 max_generations: int = 100,
                 fuzzy_rules: List[FuzzyRule] = None):
        
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.fuzzy_rules = fuzzy_rules or []
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        
    def initialize_population(self):
        """Initialize the population with random individuals"""
        self.population = []
        for _ in range(self.population_size):
            chromosome = [random.uniform(0, 1) for _ in range(self.chromosome_length)]
            individual = Individual(chromosome)
            self.population.append(individual)
    
    def evaluate_fitness(self, individual: Individual, problem_type: str = "reachability") -> float:
        """
        Evaluate fitness of an individual based on problem type
        """
        if problem_type == "reachability":
            return self._evaluate_reachability_fitness(individual)
        elif problem_type == "deadlock":
            return self._evaluate_deadlock_fitness(individual)
        else:
            return self._evaluate_general_fitness(individual)
    
    def _evaluate_reachability_fitness(self, individual: Individual) -> float:
        """
        Evaluate fitness for reachability verification problem
        Higher fitness means better reachability coverage
        """
        # Simulate reachability analysis
        reachable_states = 0
        total_states = len(individual.chromosome)
        
        for i, gene in enumerate(individual.chromosome):
            # Use fuzzy rules to determine if state is reachable
            fuzzy_inputs = [gene, i / total_states]
            reachability_score = 0.0
            
            for rule in self.fuzzy_rules:
                if len(rule.antecedents) >= len(fuzzy_inputs):
                    firing_strength = rule.evaluate(fuzzy_inputs)
                    reachability_score += firing_strength * rule.consequent.membership(gene)
            
            if reachability_score > 0.5:  # Threshold for reachability
                reachable_states += 1
        
        # Fitness is the ratio of reachable states
        fitness = reachable_states / total_states if total_states > 0 else 0.0
        
        # Add penalty for invalid configurations
        penalty = self._calculate_penalty(individual)
        
        return max(0.0, fitness - penalty)
    
    def _evaluate_deadlock_fitness(self, individual: Individual) -> float:
        """
        Evaluate fitness for deadlock detection problem
        Higher fitness means better deadlock detection capability
        """
        # Simulate deadlock detection analysis
        deadlock_probability = 0.0
        
        for i, gene in enumerate(individual.chromosome):
            # Use fuzzy rules to determine deadlock probability
            fuzzy_inputs = [gene, i / len(individual.chromosome)]
            
            for rule in self.fuzzy_rules:
                if len(rule.antecedents) >= len(fuzzy_inputs):
                    firing_strength = rule.evaluate(fuzzy_inputs)
                    deadlock_probability += firing_strength * rule.consequent.membership(gene)
        
        # Normalize deadlock probability
        deadlock_probability = min(1.0, deadlock_probability / len(self.fuzzy_rules)) if self.fuzzy_rules else 0.0
        
        # Fitness is inverse of deadlock probability (we want to avoid deadlocks)
        fitness = 1.0 - deadlock_probability
        
        # Add penalty for invalid configurations
        penalty = self._calculate_penalty(individual)
        
        return max(0.0, fitness - penalty)
    
    def _evaluate_general_fitness(self, individual: Individual) -> float:
        """General fitness evaluation"""
        # Simple fitness based on chromosome diversity and constraints
        diversity_score = len(set(individual.chromosome)) / len(individual.chromosome)
        constraint_violation = self._calculate_penalty(individual)
        
        return diversity_score - constraint_violation
    
    def _calculate_penalty(self, individual: Individual) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        # Penalty for values outside [0, 1] range
        for gene in individual.chromosome:
            if gene < 0 or gene > 1:
                penalty += abs(gene - max(0, min(1, gene)))
        
        # Penalty for extreme values
        for gene in individual.chromosome:
            if gene < 0.1 or gene > 0.9:
                penalty += 0.1
        
        return penalty
    
    def selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        
        return Individual(child1_chromosome), Individual(child2_chromosome)
    
    def mutation(self, individual: Individual):
        """Gaussian mutation with fuzzy constraints"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # Apply fuzzy mutation
                old_value = individual[i]
                
                # Gaussian mutation
                mutation_strength = 0.1
                new_value = old_value + random.gauss(0, mutation_strength)
                
                # Apply fuzzy constraints
                new_value = self._apply_fuzzy_constraints(new_value, i)
                
                individual[i] = max(0.0, min(1.0, new_value))
    
    def _apply_fuzzy_constraints(self, value: float, position: int) -> float:
        """Apply fuzzy constraints to mutation"""
        # Use fuzzy rules to constrain mutations
        if not self.fuzzy_rules:
            return value
        
        # Calculate constraint strength based on fuzzy rules
        constraint_strength = 0.0
        fuzzy_inputs = [value, position / self.chromosome_length]
        
        for rule in self.fuzzy_rules:
            if len(rule.antecedents) >= len(fuzzy_inputs):
                firing_strength = rule.evaluate(fuzzy_inputs)
                constraint_strength += firing_strength
        
        # Apply constraint
        if constraint_strength > 0.5:
            # Strong constraint - limit mutation
            return value * 0.5
        else:
            # Weak constraint - allow more mutation
            return value
    
    def evolve(self, problem_type: str = "reachability") -> Individual:
        """
        Main evolution loop
        """
        self.initialize_population()
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            for individual in self.population:
                individual.fitness = self.evaluate_fitness(individual, problem_type)
                individual.age += 1
            
            # Update best individual
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
            
            # Record fitness history
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            self.fitness_history.append(avg_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            if self.best_individual:
                new_population.append(Individual(self.best_individual.chromosome.copy(), 
                                               self.best_individual.fitness))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutation(child1)
                self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Replace population
            self.population = new_population[:self.population_size]
            
            # Early stopping if convergence
            if len(self.fitness_history) > 10:
                recent_improvement = max(self.fitness_history[-10:]) - min(self.fitness_history[-10:])
                if recent_improvement < 0.001:
                    break
        
        return self.best_individual
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get convergence metrics"""
        if not self.fitness_history:
            return {}
        
        return {
            'final_fitness': self.fitness_history[-1],
            'best_fitness': max(self.fitness_history),
            'improvement': self.fitness_history[-1] - self.fitness_history[0],
            'convergence_rate': len([i for i in range(1, len(self.fitness_history)) 
                                   if self.fitness_history[i] > self.fitness_history[i-1]]) / len(self.fitness_history)
        }


def create_default_fuzzy_rules() -> List[FuzzyRule]:
    """Create default fuzzy rules for the algorithm"""
    rules = []
    
    # Rule 1: High reachability for medium values
    rule1 = FuzzyRule(
        antecedents=[FuzzySet(0.3, 0.5, 0.7)],
        consequent=FuzzySet(0.6, 0.8, 1.0)
    )
    
    # Rule 2: Low deadlock probability for low values
    rule2 = FuzzyRule(
        antecedents=[FuzzySet(0.0, 0.2, 0.4)],
        consequent=FuzzySet(0.0, 0.2, 0.4)
    )
    
    # Rule 3: High deadlock probability for high values
    rule3 = FuzzyRule(
        antecedents=[FuzzySet(0.6, 0.8, 1.0)],
        consequent=FuzzySet(0.6, 0.8, 1.0)
    )
    
    rules.extend([rule1, rule2, rule3])
    return rules
