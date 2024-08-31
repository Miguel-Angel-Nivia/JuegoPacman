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

import util

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

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
