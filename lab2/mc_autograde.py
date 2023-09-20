import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        
        probs = []

        for state, action in zip(states, actions):

            if state[0] >= 20:

                if action == 0:
                    probs.append(1)

                else:
                    probs.append(0)
            else:

                if action == 0:
                    probs.append(0)
                    
                else:
                    probs.append(1)
        
        # End Code
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        
        if state[0] >= 20:
            action = 0

        else:
            action = 1  
            
        # End Code
        return action

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    # Keeps track of current V and count of returns for each state
    # to calculate an update.   
    V = defaultdict(float)
    returns_count = defaultdict(float)

    for episode in tqdm(range(num_episodes)): 
        
        states, actions, rewards, _ = sampling_function(env, behavior_policy)
        
        G = 0
        W = 1

        
        for t in reversed(range(len(states))):
            state, action, reward = states[t], actions[t], rewards[t]
            
            # Update total return
            G = discount_factor * G + reward
            returns_count[state] += 1

            # Here, updating W before V as per your hint
            W *= target_policy[state][action] / (behavior_policy[state][action])

            # If behavior policy is zero for taken action, we can't use this data for off-policy learning
            if behavior_policy[state][action] == 0:
                break
            
            # Value update step using the formula: Vn = Vn−1 + (1/n) * (Wn * Gn − Vn−1)
            V[state] += (W * G - V)

            
    return V
        
