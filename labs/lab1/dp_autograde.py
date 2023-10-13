import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        print(delta)
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        print(f'state {s}: delta {delta}')
        delta = max(delta, np.abs(v - V[s]))
        V[s] = vf
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        print(f'state {s}: delta {delta}')
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        print(f'state {s}+1: delta {delta}')
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    delta = 0
    for s in range(env.nS):
        v = 0
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in env.P[s][a]:
                v += action_prob * prob * (reward + discount_factor * V[next_state])
        print(f'state {s+1}: delta {delta}')
        delta = max(delta, np.abs(v - V[s]))
        V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[state]))
            V_new[state] = v
        if delta < theta:
            break
        V = V_new   
        return np.array(V)
        

            

        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    # while True:
    #     delta = 0
    #     V_new = np.zeros(env.nS)
    #     for state in range(env.nS):
    #         v = 0
    #         for action, action_prob in enumerate(policy[state]):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 v += action_prob * prob * (reward + discount_factor * V[next_state])
    #         delta = max(delta, np.abs(v - V[state]))
    #         V_new[state] = v
    #     if delta < theta:
    #         break
    #     V = V_new   
    #     return np.array(V)
        

            

    V = np.zeros(env.nS)
    while True:
      delta = 0
      for state in env.P:
        v_old = V[state]
        v_new = 0.0
        for action in env.P[state]:
          q_value = 0.0
          for transition_tuple in env.P[state][action]:
            prob, next_state, reward, done = transition_tuple
            q_value += prob * (reward + discount_factor * V[next_state])
          v_new += policy[state][action] * q_value
        V[state] = v_new
        delta = np.maximum(delta, np.abs(v_new - v_old))
      if delta < theta:
        break
    return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            v = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V_new[state] = v
            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break
        V = V_new   
        return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            v = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V_new[state] = v
        delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break
        V = V_new   
        return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            v = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V_new[state] = v
          delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break
        V = V_new   
        return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            v = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V_new[state] = v
            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break
        V = V_new   
        return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            v_old = V[state]
            v_new = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v_new += action_prob * (prob * (reward + discount_factor * V[next_state]))
              V[state] = v_new
            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break

    return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    # return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            v_old = V[state]
            v_new = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v_new += action_prob * (prob * (reward + discount_factor * V[next_state]))
              V[state] = v_new
            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break

    return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    # return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            v_old = V[state]
            v_new = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v_new += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V[state] = v_new
            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            break

    return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    # return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            v_old = V[state]
            v_new = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v_new += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V[state] = v_new
            delta = max(delta, np.abs(v_new - v_old))
        if delta < theta:
            break

    return np.array(V)
        

            

    # V = np.zeros(env.nS)
    # while True:
    #   delta = 0
    #   for state in env.P:
    #     v_old = V[state]
    #     v_new = 0.0
    #     for action in env.P[state]:
    #       q_value = 0.0
    #       for transition_tuple in env.P[state][action]:
    #         prob, next_state, reward, done = transition_tuple
    #         q_value += prob * (reward + discount_factor * V[next_state])
    #       v_new += policy[state][action] * q_value
    #     V[state] = v_new
    #     delta = np.maximum(delta, np.abs(v_new - v_old))
    #   if delta < theta:
    #     break
    # return np.array(V)        





    # for s in range(env.nS):
    #     v = 0
    #     for a, action_prob in enumerate(policy[s]):
    #         for prob, next_state, reward, done in env.P[s][a]:
    #             v += action_prob * prob * (reward + discount_factor * V[next_state])
    #     print(f'state {s+1}: delta {delta}')
    #     delta = max(delta, np.abs(v - V[s]))
    #     V[s] = v
    

    # raise NotImplementedError
    # return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # print(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            v_old = V[state]
            v_new = 0.0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v_new += action_prob * (prob * (reward + discount_factor * V[next_state]))
            V[state] = v_new
            delta = max(delta, np.abs(v_new - v_old))
        if delta < theta:
            break

    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1


            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
        
            if policy_stable:
                return policy, V
            else:
                policy[state] = policy_new

    
    # raise NotImplementedError
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    # policy = np.ones([env.nS, env.nA]) / env.nA
    # # YOUR CODE HERE
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         # compute action value function whic is Q
    #         Q = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 Q[action] += prob * (reward + discount_factor * V[next_state])
    #         policy_new = np.zeros(env.nA)
    #         policy_new[np.argmax(Q)] = 1


    #         if not np.array_equal(policy[state], policy_new):
    #             policy_stable = False
        
    #         if policy_stable:
    #             return policy, V
    #         else:
    #             policy[state] = policy_new

    
    # raise NotImplementedError
    # return policy, V
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        
        V = policy_eval(policy, env, discount_factor)
        policy_stable = True
        for state in env.P:
          old_action = policy[state]
          new_policies = np.zeros(env.nA)
          for action in env.P[state]:
            q_value = 0.0
            for transition_tuple in env.P[state][action]:
              prob, next_state, reward, done = transition_tuple
              q_value += prob * (reward + discount_factor * V[next_state])
            new_policies[action] = q_value
          policy[state] *= 0.0
          policy[state][np.argmax(new_policies)] = 1
          if np.any(policy[state] != old_action):
            policy_stable = false
        if policy_stable:
          break
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    # policy = np.ones([env.nS, env.nA]) / env.nA
    # # YOUR CODE HERE
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         # compute action value function whic is Q
    #         Q = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 Q[action] += prob * (reward + discount_factor * V[next_state])
    #         policy_new = np.zeros(env.nA)
    #         policy_new[np.argmax(Q)] = 1


    #         if not np.array_equal(policy[state], policy_new):
    #             policy_stable = False
        
    #         if policy_stable:
    #             return policy, V
    #         else:
    #             policy[state] = policy_new

    
    # raise NotImplementedError
    # return policy, V
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        
        V = policy_eval(policy, env, discount_factor)
        policy_stable = True
        for state in env.P:
          old_action = policy[state]
          new_policies = np.zeros(env.nA)
          for action in env.P[state]:
            q_value = 0.0
            for transition_tuple in env.P[state][action]:
              prob, next_state, reward, done = transition_tuple
              q_value += prob * (reward + discount_factor * V[next_state])
            new_policies[action] = q_value
          policy[state] *= 0.0
          policy[state][np.argmax(new_policies)] = 1
          if np.any(policy[state] != old_action):
            policy_stable = false
        if policy_stable:
          break
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    # policy = np.ones([env.nS, env.nA]) / env.nA
    # # YOUR CODE HERE
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         # compute action value function whic is Q
    #         Q = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 Q[action] += prob * (reward + discount_factor * V[next_state])
    #         policy_new = np.zeros(env.nA)
    #         policy_new[np.argmax(Q)] = 1


    #         if not np.array_equal(policy[state], policy_new):
    #             policy_stable = False
        
    #         if policy_stable:
    #             return policy, V
    #         else:
    #             policy[state] = policy_new

    
    # raise NotImplementedError
    # return policy, V
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in env.P:
          old_action = policy[state]
          new_policies = np.zeros(env.nA)
          for action in env.P[state]:
            q_value = 0.0
            for transition_tuple in env.P[state][action]:
              prob, next_state, reward, done = transition_tuple
              q_value += prob * (reward + discount_factor * V[next_state])
            new_policies[action] = q_value
          policy[state] *= 0.0
          policy[state][np.argmax(new_policies)] = 1
          if np.any(policy[state] != old_action):
            policy_stable = false
        if policy_stable:
          break
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1


            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
        
            if policy_stable:
                break
            else:
                policy[state] = policy_new
  return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1


            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
        
            if policy_stable:
                break
            else:
                policy[state] = policy_new
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                
            policy[state] = policy_new
        
        if policy_stable:
            break
    
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
            policy[state] *= 0.0 
            policy[state] = policy_new
        
        if policy_stable:
            break
    
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
                # policy_new = Q
            policy[state] *= 0.0 
            policy[state][np.argmax(Q)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False

        
        if policy_stable:
            break
    
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
                policy_new = Q
            policy[state] *= 0.0 
            policy[state][np.argmax(Q)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False

        
        if policy_stable:
            break
    
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            old_action = policy[state]
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] *= 0.0 
            policy[state][np.argmax(Q)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            old_action = policy[state]
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] *= 0.0 
            policy[state][np.argmax(Q)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] *= 0.0 
            policy[state][np.argmax(Q)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policiy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] *= 0.0 
            policy[state][np.argmax(policy_new)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] *= 0.0 
            policy[state][np.argmax(policy_new)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            # policy[state] *= 0.0 
            policy[state][np.argmax(policy_new)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] = np.zeros(env.nA)
            policy[state][np.argmax(policy_new)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False

        
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    policy = np.ones((env.nS, env.nA))
    V = policy_eval_v(policy, env, discount_factor)

    converged = False
    while not converged:
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] = np.zeros(env.nA)
            policy[state][np.argmax(policy_new)] = 1
            Q[state] = policy_new
            if not np.array_equal(policy[state], old_action):
                converged = False
        if converged:
            break
    policy = np.eye(env.nA)[np.argmax(Q, axis=1)]
    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    policy = np.ones((env.nS, env.nA))
    V = policy_eval_v(policy, env, discount_factor)

    
    while True:
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] = np.zeros(env.nA)
            policy[state][np.argmax(policy_new)] = 1
            Q[state] = policy_new
            if not np.array_equal(policy[state], old_action):
                policy_stable = False
        if policy_stable:
            break
    policy = np.eye(env.nA)[np.argmax(Q, axis=1)]
    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    policy = np.ones((env.nS, env.nA))
    V = policy_eval_v(policy, env, discount_factor)

    
    converged = False
    while not converged:
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] = np.zeros(env.nA)
            policy[state][np.argmax(policy_new)] = 1
            Q[state] = policy_new
            if not np.array_equal(policy[state], old_action):
                V = policy_eval_v(policy, env, discount_factor)  # update V after policy improvement
        if np.array_equal(policy, old_policy):
            converged = True
        old_policy = np.copy(policy)

    policy = np.eye(env.nA)[np.argmax(policy, axis=1)]

    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    policy = np.ones((env.nS, env.nA))
    V = policy_eval_v(policy, env, discount_factor)

    
    converged = False
    while not converged:
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] = np.zeros(env.nA)
            policy[state][np.argmax(policy_new)] = 1
            Q[state] = policy_new
            if not np.array_equal(policy[state], old_action):
                V = policy_eval_v(policy, env, discount_factor)  # update V after policy improvement
        if np.array_equal(policy, Q):
            converged = True
        old_policy = np.copy(policy)

    policy = np.eye(env.nA)[np.argmax(policy, axis=1)]

    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            q = 0.0
            for action in range(env.nA):
                

    

    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            for action in range(env.nA):
                q = 0.0
                for prob, next_state, reward, done in env.P[state][action]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                delta = max(delta, np.abs(q - Q[state][action]))
                Q[state][action] = q
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        policy[state][np.argmax(Q[state])] = 1
    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            for action in range(env.nA):
                q = 0.0
                q_old = Q[state][action]
                for prob, next_state, reward, done in env.P[state][action]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                delta = max(delta, np.abs(q - q_old))
                Q[state][action] = q
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        policy[state][np.argmax(Q[state])] = 1
    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            for action in range(env.nA):
                q = 0.0
                q_old = Q[state][action]
                for prob, next_state, reward, done in env.P[state][action]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                delta = max(delta, np.abs(q - q_old))
            Q[state][action] = q
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        policy[state][np.argmax(Q[state])] = 1
    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            for action in range(env.nA):
                q = 0.0
                q_old = Q[state][action]
                for prob, next_state, reward, done in env.P[state][action]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                delta = max(delta, np.abs(q - q_old))
                Q[state][action] = q
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        policy[state][np.argmax(Q[state])] = 1
    # raise NotImplementedError
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    while True:
        delta = 0
        for state in range(env.nS):
            for action in range(env.nA):
                q = 0.0
                q_old = Q[state][action]
                for prob, next_state, reward, done in env.P[state][action]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                Q[state][action] = q
                delta = max(delta, np.abs(q - q_old))
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        policy[state][np.argmax(Q[state])] = 1
    # raise NotImplementedError
    return policy, Q

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         old_action = policy[state]
    #         policy_new = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 policy_new[action] += prob * (reward + discount_factor * V[next_state])
    #         policy[state] = np.zeros(env.nA)
    #         policy[state][np.argmax(policy_new)] = 1
    #         if not np.array_equal(policy[state], old_action):
    #             policy_stable = False
    #     if policy_stable:
    #         break

    while True:

        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            old_action = policy[state]
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy[state] = np.zeros(env.nA)
            policy[state][np.argmax(policy_new)] = 1
            if not np.array_equal(policy[state], old_action):
                policy_stable = False
        if policy_stable:
            break
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE


    while True:

        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            action = np.argmax(policy[state])
            for action in range(env.nA):
                action_prob = policy[state][action]
                policy[state][action] = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    policy[state][action] += prob * (reward + discount_factor * V[next_state])
            best_action = np.argmax(policy[state])
        if policy[state][best_action] != 1:
            policy_stable = False
        if policy_stable:
            break


    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1


            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
        
            if policy_stable:
                return policy, V
            else:
                policy[state] = policy_new

    
    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    # policy = np.ones([env.nS, env.nA]) / env.nA
    # # YOUR CODE HERE
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         # compute action value function whic is Q
    #         Q = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 Q[action] += prob * (reward + discount_factor * V[next_state])
    #         policy_new = np.zeros(env.nA)
    #         policy_new[np.argmax(Q)] = 1


    #         if not np.array_equal(policy[state], policy_new):
    #             policy_stable = False
        
    #         if policy_stable:
    #             return policy, V
    #         else:
    #             policy[state] = policy_new

    
    # raise NotImplementedError
    # return policy, V
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        
        V = policy_eval(policy, env, discount_factor)
        policy_stable = True
        for state in env.P:
          old_action = policy[state]
          new_policies = np.zeros(env.nA)
          for action in env.P[state]:
            q_value = 0.0
            for transition_tuple in env.P[state][action]:
              prob, next_state, reward, done = transition_tuple
              q_value += prob * (reward + discount_factor * V[next_state])
            new_policies[action] = q_value
          policy[state] *= 0.0
          policy[state][np.argmax(new_policies)] = 1
          if np.any(policy[state] != old_action):
            policy_stable = false
        if policy_stable:
          break
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
        
            if policy_stable:
                break
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
        
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in env.P:
    #       old_action = policy[state]
    #       new_policies = np.zeros(env.nA)
    #       for action in env.P[state]:
    #         q_value = 0.0
    #         for transition_tuple in env.P[state][action]:
    #           prob, next_state, reward, done = transition_tuple
    #           q_value += prob * (reward + discount_factor * V[next_state])
    #         new_policies[action] = q_value
    #       policy[state] *= 0.0
    #       policy[state][np.argmax(new_policies)] = 1
    #       if np.any(policy[state] != old_action):
    #         policy_stable = false
    #     if policy_stable:
    #       break
    
    # return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            # compute action value function whic is Q
            Q = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    Q[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(Q)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        
            if policy_stable:
                break
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         old_action = policy[state]
    #         policy_new = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 policy_new[action] += prob * (reward + discount_factor * V[next_state])
    #         policy[state] = np.zeros(env.nA)
    #         policy[state][np.argmax(policy_new)] = 1
    #         if not np.array_equal(policy[state], old_action):
    #             policy_stable = False
    #     if policy_stable:
    #         break

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            
            new_policy = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    new_policy[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(new_policy)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        
            if policy_stable:
                break
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         old_action = policy[state]
    #         policy_new = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 policy_new[action] += prob * (reward + discount_factor * V[next_state])
    #         policy[state] = np.zeros(env.nA)
    #         policy[state][np.argmax(policy_new)] = 1
    #         if not np.array_equal(policy[state], old_action):
    #             policy_stable = False
    #     if policy_stable:
    #         break

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            new_policy = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    new_policy[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(new_policy)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        if policy_stable:
            break
    return policy, V

    
    # raise NotImplementedError
    # return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         old_action = policy[state]
    #         policy_new = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 policy_new[action] += prob * (reward + discount_factor * V[next_state])
    #         policy[state] = np.zeros(env.nA)
    #         policy[state][np.argmax(policy_new)] = 1
    #         if not np.array_equal(policy[state], old_action):
    #             policy_stable = False
    #     if policy_stable:
    #         break

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            new_policy = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    new_policy[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(new_policy)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        if policy_stable:
            break


    # raise NotImplementedError
    return policy, V
    # policy = np.ones([env.nS, env.nA]) / env.nA
    
    # while True:
    #     V = policy_eval_v(policy, env, discount_factor)
    #     policy_stable = True
    #     for state in range(env.nS):
    #         old_action = policy[state]
    #         policy_new = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             for prob, next_state, reward, done in env.P[state][action]:
    #                 policy_new[action] += prob * (reward + discount_factor * V[next_state])
    #         policy[state] = np.zeros(env.nA)
    #         policy[state][np.argmax(policy_new)] = 1
    #         if not np.array_equal(policy[state], old_action):
    #             policy_stable = False
    #     if policy_stable:
    #         break

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            policy_new = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    policy_new[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(policy_new)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        if policy_stable:
            break


    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            new_policy = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    new_policy[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(new_policy)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        if policy_stable:
            break


    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            naction_values = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(action_values)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        if policy_stable:
            break


    # raise NotImplementedError
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True
        for state in range(env.nS):
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += prob * (reward + discount_factor * V[next_state])
            policy_new = np.zeros(env.nA)
            policy_new[np.argmax(action_values)] = 1
            if not np.array_equal(policy[state], policy_new):
                policy_stable = False
                policy[state] = policy_new 
        if policy_stable:
            break


    # raise NotImplementedError
    return policy, V
