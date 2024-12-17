import numpy as np
from typing import Literal, Dict, Hashable
import json
import matplotlib.pyplot as plt

ValueMethods = Literal['value_iteration', 'monte_carlo', 'temporal_difference']
PolicyMethods = Literal['policy_iteration', 'value_iteration', 'q_learning']
StateType = Hashable
ActionType = Hashable

PolicyType = Dict[StateType, ActionType 
                            | Dict[ActionType, float] ]

np.random.seed(42)

class HomoMDP:

  def __init__(self, 
              initial_distribution,
              responses, 
              name = "MDP", 
              nonterminal_states = None, 
              terminal_states = None,
              actions = None, ):
    """
      MDP constructor
      ----------
      initial_distribution : {[state]: probability}
      responses: {[state, action] : (next_state, probability, reward)}
      name : str
      nonterminal_states : list(), optional. If None, nonterminal states will be inferred from the keys of responses.
      terminal_states: list(), optional. If None, terminal states will be inferred from states not appearing in the keys of responses.
      actions: list(), optional. If None, actions will be inferred from the keys of responses.
    """

    self.initial_distribution = initial_distribution

    self.responses = dict()
    for state, action in responses.keys():
      if state not in self.responses:
        self.responses[state] = {}
      self.responses[state][action] = responses[state, action]

    self.nonterminal_states = nonterminal_states \
                            if nonterminal_states is not None \
                            else list(set(state for state, _ in responses.keys()))
        
    self.terminal_states = terminal_states \
                            if terminal_states is not None \
                            else list({state for response in responses.values() for state, _, _ in response} 
                                      - set(self.nonterminal_states))
    
    self.states = self.nonterminal_states + self.terminal_states

    self.actions = actions if actions is not None else list(set(action for _, action in responses.keys()))
    
    self.name = name

    self.valid_actions_dict = {state: [] for state in self.nonterminal_states}
    for state, action in responses.keys():
      self.valid_actions_dict[state].append(action)

  def reset(self):
    """
      Get initial state from initial distribution
    """

    return HomoMDP.__sample_from_distribution(self.initial_distribution)
  
  def best_policy(self, 
                  method:PolicyMethods='policy_iteration', 
                  discount=1, 
                  explore=0,
                  step_size=0.1,
                  eps=1e-3, 
                  max_iter=50, 
                  log=False, plot=False):
    if method == 'policy_iteration': 
      policy = {state: np.random.choice(list(self.responses[state].keys()))
                for state in self.nonterminal_states}
      
      if log:
        print("Initial policy", policy)

      for i in range(max_iter):
        # Evaluate policy
        V = self.policy_evaluation(policy, 
                                    discount=discount, 
                                    eps=eps, 
                                    max_iter=max_iter)
        # Improve policy
        best_actions = self.__best_actions(V, discount)

        if policy == best_actions:
          print(f"Best policy: converged after {i} iterations")
          break

        policy = best_actions

        if log:
          print(policy)

        if i == max_iter-1:
          print("Best policy: max iterations reached")

      return policy

    elif method == 'value_iteration':
      V = {state : 0 for state in self.states}

      for i in range(max_iter):
        diff = 0
        new_values = {state : 0 for state in self.states}
        for state in self.nonterminal_states:
          value = -np.inf

          for action in self.valid_actions_dict[state]:
            action_value = self.__update_value(state, action, V, discount)
            if action_value > value:
              new_values[state] = action_value
        
        diff = np.max([max(diff, abs(new_values[state] - V[state])) 
                       for state in self.states])
        if diff < eps:
          print(f"Value iteration: converged after {i} iterations")
          break
        
        if i == max_iter-1:
          print("Value iteration: max iterations reached")

        V = new_values

      print("Last value", V)

      return self.__best_actions(V, discount)

    elif method == 'q_learning':
      Q = {state: {action: np.random.rand() for action in self.valid_actions_dict[state]} 
                              for state in self.nonterminal_states}
      for state in self.terminal_states:
        Q[state] = dict()
        for action in self.actions:
          Q[state][action] = 0

      total_reward_history = []

      for iter in range(1, max_iter+1):
        current_state = self.reset()
        diff = 0
        total_reward = 0
        history = [current_state]
        while current_state not in self.terminal_states:
          action = self.__best_action(current_state, Q=Q)
          
          if np.random.rand() < explore:
            action = np.random.choice(list(self.responses[current_state].keys()))

          next_state, reward = self.__observe(current_state, action)
          q_new = Q[current_state][action] + step_size*(reward + discount*np.max(list(Q[next_state].values())) - Q[current_state][action])

          diff = max(diff, abs(q_new - Q[current_state][action]))
          Q[current_state][action] = q_new
          history.append((current_state, action, reward))
          current_state = next_state
          total_reward += reward
        
        best_actions = self.__best_actions(Q=Q)
        total_reward_history.append(total_reward)
        
        # if log:
        #   print(f"Trajectory {iter}: {history}")

        
        # if diff < eps:
        #   print(f"Q-Learning converges in {iter} iterations")
        #   break

        # if iter == max_iter:
        #   print(f"Q-Learning reached max iterations")
      
      if log:
        print("Final Q:")
        print(json.dumps({str(k): v for k, v in Q.items()}, indent=2))
      
      if plot:
        plt.plot(total_reward_history)
        plt.xlabel('Iteration')
        plt.ylabel('Total reward')

        plt.show()



      return self.__best_actions(Q=Q)
    else:
      raise ValueError(f"Valid methods are {PolicyMethods}")

  def policy_evaluation(self, 
                        policy: PolicyType,     
                        method: ValueMethods='value_iteration', 
                        discount=1, 
                        step_size = 0.1,
                        _lambda = 0,
                        eps=1e-4, 
                        max_iter=50,
                        log=False):
    V = {state: 0 for state in self.states}
    
    if method == 'value_iteration':
      for i in range(max_iter):
        diff = 0

        for state, action in policy.items():
          if state in self.terminal_states:
            continue
          
          new_value = 0
          
          response = self.responses[state].get(action)

          if response is None:
            raise ValueError(f"Action {action} is not valid state {state}")
          
          for next_state, prob, reward in response:
            new_value += prob * (reward + discount*V[next_state])

          diff = max(diff, abs(new_value - V[state]))
          V[state] = new_value
        
        if diff < eps and log:
          print(f"Policy evaluation: converged after {i} iterations")
          break

        if i == max_iter-1 and log:
          print("Policy evaluation: max iteration reached")
    elif method == 'temporal_difference':
      V = self.temporal_difference(policy, _lambda, step_size, discount, eps, max_iter)

    return V

  def __best_actions(self, V=None, Q=None, discount=1):
    if V is None and Q is None:
      raise ValueError("Either V or Q must be provided")
    
    if V is not None:
      return {state: self.__best_action(state, V=V, discount=discount) for state in self.nonterminal_states}
    
    return {state: self.__best_action(state, Q=Q, discount=discount) for state in self.nonterminal_states}
  
  def __best_action(self, state, V=None, Q=None, discount=1):
    if V is None and Q is None:
      raise ValueError("Either V or Q must be provided")
    
    if V is not None:
      action_values = {}
      for action, response in self.responses[state].items():
        action_values[action] = np.sum([prob * (reward + discount*V[next_state]) for next_state, prob, reward in response])
      
      return list(action_values.keys())[np.argmax(list(action_values.values()))]
    
    if Q is not None:    
      return list(Q[state].keys())[np.argmax(list(Q[state].values()))]

  def __update_value(self, state, action, V, discount=1):
    response = self.responses[state][action]
    value = 0
    
    for next_state, prob, reward in response:
      value += prob * (reward + discount*V[next_state])
    
    return value
  
  def temporal_difference(self, policy, 
                          _lambda = 0, 
                          step_size = 0.1,
                          discount = 1,
                          eps=1e-3, 
                          max_iter=1000):
    V = {state : 0 for state in self.terminal_states + self.nonterminal_states}

    if _lambda == 0:
      diff = 0

      for iter in range(1, max_iter+1):
        current_state = self.reset()

        while current_state not in self.terminal_states:
          next_state, reward = self.__observe(current_state, policy[current_state])

          new_value = V[current_state] + step_size*(reward + discount*V[next_state] - V[current_state])

          diff = max(diff, abs(new_value - V[current_state]))
          V[current_state] = new_value
          current_state = next_state
        
        if diff < eps:
          print(f"TD({_lambda}) converged after {iter} iterations.")
          break
      
        if iter == max_iter:
          print(f"TD({_lambda}) reached maximal iterations.")

      return V

  def __observe(self, current_state, action):
    response_dict = dict()
    for next_state, prob, reward in self.responses[current_state][action]:
      response_dict[next_state, reward] = prob

    return HomoMDP.__sample_from_distribution(response_dict)

  def total_reward(self, policy):
    return np.sum([self.expected_reward(state, action) for state, action in policy.items()])

  def expected_reward(self, state, action):
    return np.sum([prob * reward for _, prob, reward in self.responses[state][action]])

  @staticmethod
  def __sample_from_distribution(distribution : dict):
    keys = list(distribution.keys())

    index = np.random.choice(range(0, len(keys)), p=list(distribution.values()))

    return keys[index]