# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
from game import Actions
from pacman import GameState
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        #self.qValues = [['North', 0], ['East', 0], ['South', 0], ['West', 0]]
        # Local copy of the game score
        self.scoreTracker = 0
        # How long have we been running?
        self.steps = 0
        # qValues for each possible action and state pair
        self.qValues = {}

    def setScoreTracker(self, score):
        self.scoreTracker = score

    def getScoreTracker(self):
        return self.scoreTracker
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getEpsilon(self):
        return self.epsilon

    def getMaxAttempts(self):
        return self.maxAttempts

    # Return the index of the highest qValue among the values in the list
    def selectQlearnAction(self, list):
        index = random.randint(0, (len(list) - 1))
        max = list[index]
        for i in range(len(list)):
            if list[i] > max:
                index = i
                max =  list[i]
        return index

    # Get the previous state of a given state-action pair
    def getLastState(self, last_action, current_state):
        last_state = current_state
        if last_action == 'East':
            last_state = tuple([(current_state[0] - 1), current_state[1]])
        elif last_action == 'West':
            last_state = tuple([(current_state[0] + 1), current_state[1]])
        elif last_action == 'North':
            last_state = tuple([current_state[1],(current_state[1] - 1)])
        else:
            last_state = tuple([current_state[0],(current_state[1] + 1)])
        return last_state

    # Get the maximum qValues among the list of qValues for the state
    # Return 0 if the state is unseen
    def getQmax(self, state):
        q_max = 0
        q_values = []
        for each_key in self.qValues.keys():
            if state in each_key:
                q_values.append(self.qValues[each_key])
        if len(q_values) != 0:
            q_max = max(q_values)
            return q_max
        else:
            return 0.0

    # Function for updating the qValues using the Qlearning update rule
    def updateQValue(self, action, state, nextState, value):
        if action != 'Stop':
            q_max = self.getQmax(nextState)
            if tuple([state, action]) in self.qValues.keys():
                self.qValues[tuple([state, action])] = self.qValues[tuple([state, action])] + self.alpha*(value + self.gamma*q_max - self.qValues[tuple([state, action])])

    # Accessor function for qValues 
    # returns Q(state, action) for seen states
    def getQValue(self, state, action):
        if tuple([state,action]) in self.qValues.keys():
            return self.qValues[tuple([state,action])] 
        else:
            return 0.0 

    # Function to alter the legal actions according to the ghost's position
    def avoidGhost(self, legal, current_pos, ghost_pos):
        # find the distance between the Pacman and ghost in x and y directions
        dist_x = int(ghost_pos[0][0]) - int(current_pos[0])
        dist_y = int(ghost_pos[0][1]) - int(current_pos[1])

        # Depending upon the position of the ghost remove the action
        # from the legal action list that would cause collision with the ghost
        if dist_x == 1 and dist_y == 0:
            if Directions.EAST in legal and len(legal) != 1:
                legal.remove(Directions.EAST)
        elif dist_x == -1 and dist_y == 0:
            if Directions.WEST in legal and len(legal) != 1:
                legal.remove(Directions.WEST)
        elif dist_y == -1 and dist_x == 0:
            if Directions.SOUTH in legal and len(legal) != 1:
                legal.remove(Directions.SOUTH)
        elif dist_y == 1 and dist_x == 0:
            if Directions.NORTH in legal and len(legal) != 1:
                legal.remove(Directions.NORTH)
        return legal

    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        # The data we have about the state of the game
        # the last action performed by pacman
        last_action = state.getPacmanState().configuration.direction

        # Pacman's current position
        current_pos = state.getPacmanState().configuration.pos

        # Pacman's last position
        last_pos = self.getLastState(last_action, current_pos)

        # Ghost's current position
        ghost_pos = state.getGhostPositions()

        # Calculate the change in score due to the last action, and
        # update our local copy of the score
        current_score = state.getScore()
        change_in_score = current_score - self.getScoreTracker()
        self.setScoreTracker(current_score)
        
        # Since we are still running, we need to update the qvalues associated 
        # with the last position and action
        
        self.updateQValue(last_action, last_pos, current_pos, change_in_score)
      
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Avoid the action that would lead to collision with the ghost
        legal = self.avoidGhost(legal, current_pos, ghost_pos)

        # For each action in legal, build a list with the current q-values:
        q_values = []
        for i in range(len(legal)):
           q_values.append(self.getQValue(current_pos,legal[i]))
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:" , state.getGhostPositions()
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()

        # Make the qlearning choice
        choice = random.random()

        # Now pick what action to take. If choice is > 1-Epsilon
        # pick the qlearning action
        if choice <= (1 - self.getEpsilon()):
            actionIndex = self.selectQlearnAction(q_values)
            action = legal[actionIndex]
        else:
            # otherwise pick a random action
            action = random.choice(legal)
        # We have to return an action
        return action
    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        print "A game just ended!"
        # get the very last action made by Pacman just before the end of the game
        final_action = state.getPacmanState().configuration.direction
        # get the current position of Pacman
        current_pos = state.getPacmanState().configuration.pos
        # get the final position of Pacman-the last position before the end of the game
        final_pos =  self.getLastState(final_action, current_pos)
        # get the value of the final score
        final_score = state.getScore()

        # if Pacman wins
        if final_score > 0:
            # if the final state is an unseen state
            if tuple([final_pos, final_action]) not in self.qValues.keys():
                self.qValues[tuple([final_pos, final_action])] = 0.0 + 500.0
            else:
                self.qValues[tuple([final_pos, final_action])] = self.qValues[tuple([final_pos, final_action])] + 500.0

        # if Pacman loses
        if final_score < 0:
            # if the final state is an unseen state
            if tuple([final_pos, final_action]) not in self.qValues.keys():
                self.qValues[tuple([final_pos, final_action])] = 0.0 - 500.0
            else:
                self.qValues[tuple([final_pos, final_action])] = self.qValues[tuple([final_pos, final_action])] - 500.0
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


