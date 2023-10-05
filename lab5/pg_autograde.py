import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE

        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        probs = F.softmax(x, dim=1)

        return probs
        
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        
        probs = self.forward(obs)
        
        action_probs = probs.gather(1, actions)

        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # print(obs.shape)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
    
        
        probs = self.forward(obs)

        action_probs = probs.squeeze().detach().numpy()
        action = np.random.choice([0, 1], p=action_probs)
        
        return int(action)
        
        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    
    state = env.reset()
    done = False

    while not done:
    
        while not done:
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add a batch dimension
            states.append(state_tensor)
            
            action = policy.sample_action(state_tensor.squeeze(0))
            actions.append(action) 
            state, reward, done, _ = env.step(action)
            rewards.append([reward])  # Making it 2D
            dones.append([done])     # Making it 2D

    states = torch.cat(states, dim=0)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    return states, actions, rewards, dones

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    # YOUR CODE HERE
    
    states, actions, rewards, dones = episode
    probs = policy.get_probs(states, actions)
    returns = []

    rewards_np = np.array(rewards)
    for i in range(len(rewards)):
        discount_factors = discount_factor ** np.arange(len(rewards) - i)
        return_i = np.sum(rewards_np[i:] * discount_factors)
        returns.append(return_i)

    returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)  # Convert to tensor and add a dimension for Nx1 shape
    loss = -torch.mean(torch.log(probs) * returns)

    return loss

# YOUR CODE HERE

loss = compute_reinforce_loss(policy, trajectory_data, 0.9)
print(loss)


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        
        # YOUR CODE HERE
        
        optimizer.zero_grad()
        episode = sampling_function(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations
