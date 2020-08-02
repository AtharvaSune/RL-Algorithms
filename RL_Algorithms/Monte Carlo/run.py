from IPython import get_ipython

import gym
import matplotlib 
import numpy as np
from IPython.display import clear_output
import operator
import time
import random
import pprint

class MonteCarlo():
    """
        First Visit Monte Carlo implementation
    """
    def __init__(self,  n_states, n_actions, epsilon=None, explore=None, discount=1):
        self._policy = self._create_default_policy(n_states, n_actions)
        self._qvalues = self._initialize_qvalues(n_states, n_actions)
        self.legal_actions = list(range(0, n_actions))
        self._visit = np.zeros(n_states)

        self.explore = explore
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """
            return the qvalue of a (state, action) pair
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """
            set the qvalue of a (state, action) pair
        """
        self._qvalues[state][action] = value
    
    def _create_default_policy(self, n_states, n_actions):
        """
            create a starting policy
        """
        
        policy = {}
        for state in range(0, n_states):
            p = {}
            for action in range(0, n_actions):
                p[action] = 1/n_actions
            policy[state] = p

        return policy

    def _initialize_qvalues(self, n_states, n_actions):
        """
            create a starting qvalue dictionary
        """
        qvalues = {}
        for state in range(0, n_states):
            q = {}
            for action in range(0, n_actions):
                q[action] = 0.0
            
            qvalues[state] = q
        
        return qvalues
    
    def get_best_action(self, state):
        """ 
            return the best action for a given state
        """
        possible_actions = self.legal_actions
        return np.argmax([self._policy[state][action] for action in possible_actions])

    def select_action(self, state, random_s):
        """
            select an action when in a state
        """

        if random_s:
            return random.choice(self.legal_actions)

        if self.explore is None:
            return self.get_best_action(state)

        else:
            choice = np.random.random()
            if choice < self.explore:
                return random.choice(self.legal_actions)
            else:
                return self.get_best_action(state)

    def update(self, episode_info, reward):
        """
            update the policy after episode ends
        """
        G = 0
        Returns = {}

        # Reverse propogate the reward 
        for i in reversed(range(0,len(episode_info))):
            s, a, r = episode_info[i]
            G = self.discount*G + reward
            reward = r
            
            if not (s, a) in Returns.keys():
                Returns[(s, a)] = [G]
            else:
                Returns[(s, a)].append(G)
        
        # Set Q values
        for ((state, action), reward_list) in Returns.items():
            self.set_qvalue(state, action, sum(reward_list)/len(reward_list))

        # Update the policy
        for state in self._policy.keys():
            A_max = np.argmax([self.get_qvalue(state, action) for action in self.legal_actions])
            
            for action in self.legal_actions:
                if action == A_max:
                    self._policy[state][action] = 1 - self.epsilon + self.epsilon/len(self.legal_actions)
                else:
                    self._policy[state][action] = self.epsilon/len(self.legal_actions)


def play_and_learn(agent, env, num_episodes, pp):

    s = env.reset()
    for _ in range(num_episodes):
        if _%20==0:
            print("*"*100, end="\n\n")
            print(f"Episode: {_}")
            pp.pprint(agent._policy)
            print("-"*100)
        finished = False
        episode_info = []
        while not finished:
            action = agent.select_action(s, random_s=True)
            next_s, r, finished, __ = env.step(action)
            episode_info.append((s, action, r))
            s = next_s
        agent.update(episode_info, r)
        if _%20==0:
            pp.pprint(agent._policy)
            print("-"*100)
            print("*"*100, end="\n\n")        

def MonteCarloTraining(env, epsilon, explore, discount, num_episodes=100):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    pp = pprint.PrettyPrinter(indent=4)
    agent = MonteCarlo(env.observation_space.n, env.action_space.n, epsilon=epsilon, explore=explore, discount=discount)

    play_and_learn(agent, env, num_episodes, pp)


env = gym.make("FrozenLake8x8-v0")

MonteCarloTraining(env, 0.4, 0.6, 0.4)
# print(str(env))
