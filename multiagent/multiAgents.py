# multiAgents.py
# --------------
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

# -*- coding: utf-8 -*-
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
      # Genera el estado sucesor
      successorGameState = currentGameState.generatePacmanSuccessor(action)
      newPos = successorGameState.getPacmanPosition()
      newFood = successorGameState.getFood()
      newGhostStates = successorGameState.getGhostStates()
      newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

      # Inicializa la puntuacion
      score = successorGameState.getScore()

      # Evalua la distancia a la comida
      foodList = newFood.asList()
      if foodList:
          minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
          score += 1.0 / (minFoodDist + 1)  # Incentiva acercarse a la comida

      # Evalua la distancia a los fantasmas
      for ghostState in newGhostStates:
          ghostPos = ghostState.getPosition()
          ghostDist = manhattanDistance(newPos, ghostPos)
          if ghostDist > 0:
              if ghostState.scaredTimer > 0:
                  # Si el fantasma esta asustado, es beneficioso acercarse
                  score += 2.0 / ghostDist
              else:
                  # Si el fantasma no esta asustado, es peligroso acercarse
                  if ghostDist < 2:
                      score -= 1000  # Penalizacion grande por estar muy cerca
                  else:
                      score -= 1.0 / ghostDist

      # Incentiva comer capsulas de poder
      if currentGameState.getPacmanPosition() != newPos:
          if newPos in currentGameState.getCapsules():
              score += 150

      # Incentiva acciones que resultan en comer comida
      if newFood.count() < currentGameState.getFood().count():
          score += 200

      return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
      def getAction(self, gameState):
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if agentIndex == 0:  # Pacman's turn (MAX)
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(successor, depth, 1))
                return value
            else:  # Ghosts' turn (MIN)
                value = float('inf')
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth - 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, minimax(successor, nextDepth, nextAgent))
                return value

        # Root level action choice for Pacman
        best_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
      def alphaBeta(state, depth, agentIndex, alpha, beta):
          if depth == 0 or state.isWin() or state.isLose():
              return self.evaluationFunction(state)
          
          if agentIndex == 0:  # Pacman's turn (MAX)
              value = float('-inf')
              for action in state.getLegalActions(agentIndex):
                  successor = state.generateSuccessor(agentIndex, action)
                  value = max(value, alphaBeta(successor, depth, 1, alpha, beta))
                  if value > beta:
                      return value
                  alpha = max(alpha, value)
              return value
          else:  # Ghosts' turn (MIN)
              value = float('inf')
              nextAgent = (agentIndex + 1) % state.getNumAgents()
              nextDepth = depth - 1 if nextAgent == 0 else depth
              for action in state.getLegalActions(agentIndex):
                  successor = state.generateSuccessor(agentIndex, action)
                  value = min(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                  if value < alpha:
                      return value
                  beta = min(beta, value)
              return value

      # Root level action choice for Pacman
      best_score = float('-inf')
      best_action = None
      alpha = float('-inf')
      beta = float('inf')
      for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          score = alphaBeta(successor, self.depth, 1, alpha, beta)
          if score > best_score:
              best_score = score
              best_action = action
          alpha = max(alpha, best_score)
      return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if agentIndex == 0:  # Pacman's turn (MAX)
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, depth, 1))
                return value
            else:  # Ghosts' turn (EXP)
                value = 0
                legalActions = state.getLegalActions(agentIndex)
                probability = 1.0 / len(legalActions)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth - 1 if nextAgent == 0 else depth
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value += probability * expectimax(successor, nextDepth, nextAgent)
                return value

        # Root level action choice for Pacman
        best_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

def betterEvaluationFunction(currentGameState):
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Calculate distance to the nearest food
    foodList = newFood.asList()
    if len(foodList) > 0:
        minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        minFoodDist = 0

    # Calculate distance to the nearest ghost
    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    minGhostDist = min(ghostDistances) if ghostDistances else 0

    # Calculate features
    score = currentGameState.getScore()
    numFood = currentGameState.getNumFood()
    numCapsules = len(currentGameState.getCapsules())

    # Evaluate the state
    evaluation = score + (1.0 / (minFoodDist + 1)) - (1.0 / (minGhostDist + 1)) - (2 * numFood) - (10 * numCapsules)

    # Consider scared ghosts
    for ghostState, distance in zip(newGhostStates, ghostDistances):
        if ghostState.scaredTimer > 0:
            evaluation += (1.0 / (distance + 1)) * 100  # Attract to scared ghosts

    return evaluation

# Abbreviation
better = betterEvaluationFunction