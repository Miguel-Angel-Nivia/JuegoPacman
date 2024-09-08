# -- coding: utf-8 --
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util, random

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    # pila
    stack = util.Stack()
    # Conjunto visitados
    visited = set()
    # Añadir el estado inicial a la pila
    start_state = problem.getStartState()
    stack.push((start_state, []))
    while not stack.isEmpty():
        current_state, actions = stack.pop()
        # final objetivo
        if problem.isGoalState(current_state):
            return actions
        # Si el estado no ha sido visitado, explorarlo
        if current_state not in visited:
            visited.add(current_state)
            # Obtener los sucesores del estado actual
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    # Añadir el sucesor a la pila con las acciones actualizadas
                    new_actions = actions + [action]
                    stack.push((successor, new_actions))
    # Si no se encuentra solución, devolver una lista vacía
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    # cola
    queue = util.Queue()
    # Conjunto visitados
    visited = set()
    # Añadir el estado inicial a la cola
    start_state = problem.getStartState()
    queue.push((start_state, []))
    while not queue.isEmpty():
        current_state, actions = queue.pop()
        # fin objetivo
        if problem.isGoalState(current_state):
            return actions
        # Si el estado no ha sido visitado, explorarlo
        if current_state not in visited:
            visited.add(current_state)
            # Obtener los sucesores del estado actual
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    # Añadir el sucesor a la cola con las acciones actualizadas
                    new_actions = actions + [action]
                    queue.push((successor, new_actions))
    # Si no se encuentra solución, devolver una lista vacía
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # cola de prioridad
    pq = util.PriorityQueue()
    # Conjunto visitados
    visited = set()
    # Añadir el estado inicial a la cola de prioridad
    start_state = problem.getStartState()
    pq.push((start_state, [], 0), 0)
    while not pq.isEmpty():
        current_state, actions, current_cost = pq.pop()
        # fin objetivo
        if problem.isGoalState(current_state):
            return actions
        # Si el estado no ha sido visitado, explorarlo
        if current_state not in visited:
            visited.add(current_state)   
            # Obtener los sucesores del estado actual
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # Calcular el nuevo costo y las nuevas acciones
                    new_cost = current_cost + step_cost
                    new_actions = actions + [action]
                    # Añadir el sucesor a la cola de prioridad
                    pq.push((successor, new_actions, new_cost), new_cost) 
    # Si no se encuentra solución, devolver una lista vacía
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # cola prioridad
    pq = util.PriorityQueue()
    # Conjunto visitados
    visited = set()
    # Añadir el estado inicial a la cola de prioridad
    start_state = problem.getStartState()
    start_h = heuristic(start_state, problem)
    pq.push((start_state, [], 0), start_h)
    while not pq.isEmpty():
        current_state, actions, current_cost = pq.pop()
        # fin objetivo
        if problem.isGoalState(current_state):
            return actions
        # Si el estado no ha sido visitado, explorarlo
        if current_state not in visited:
            visited.add(current_state)
            # Obtener los sucesores del estado actual
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # Calcular el nuevo costo g (costo real hasta ahora)
                    new_cost = current_cost + step_cost
                    # Calcular el costo h (heurístico estimado hasta el objetivo)
                    h = heuristic(successor, problem)
                    # Calcular el costo f total (f = g + h)
                    f = new_cost + h
                    new_actions = actions + [action]
                    # Añadir el sucesor a la cola de prioridad
                    pq.push((successor, new_actions, new_cost), f)
    # Si no se encuentra solución, devolver una lista vacía
    return []

def geneticSearch(problem):
    """Search using genetic algorithm."""
    
    def initialize_population(size):
        return [random_actions(problem) for _ in range(size)]
    
    def random_actions(problem):
        actions = []
        state = problem.getStartState()
        while not problem.isGoalState(state):
            successors = problem.getSuccessors(state)
            if not successors:
                return None  # No solution
            state, action, _ = random.choice(successors)
            actions.append(action)
        return actions

    def evaluate_fitness(actions):
        if actions is None:
            return 0
        return 1.0 / (problem.getCostOfActions(actions) + 1)

    def select_parents(population, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        return [roulette_wheel_selection(population, selection_probs) for _ in range(len(population))]
    
    def roulette_wheel_selection(population, probabilities):
        r = random.random()
        for i, individual in enumerate(population):
            if r <= sum(probabilities[:i+1]):
                return individual
    
    def crossover(parent1, parent2):
        if parent1 is None or parent2 is None:
            return parent1 if parent2 is None else parent2
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(actions):
        if actions is None:
            return None
        if len(actions) <= 1:
            return actions
        i = random.randint(0, len(actions) - 1)
        state = problem.getStartState()
        for j in range(i):
            state, _, _ = problem.getSuccessors(state)[0]
        successors = problem.getSuccessors(state)
        if successors:
            _, new_action, _ = random.choice(successors)
            actions[i] = new_action
        return actions

    # Genetic Algorithm parameters
    population_size = 5
    generations = 5
    mutation_rate = 0.1

    # Initialize population
    population = initialize_population(population_size)

    for _ in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Select parents
        parents = select_parents(population, fitness_scores)
        
        # Create new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        
        population = new_population

    # Return best solution
    best_individual = max(population, key=evaluate_fitness)
    # print(best_individual)  # Visualizacion tiempo y ruta
    return best_individual

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
gen = geneticSearch