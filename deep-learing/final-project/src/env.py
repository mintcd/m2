import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class TaskSchedulingEnv(gym.Env):
    def __init__(self, N_total_nodes, total_tasks):
        super(TaskSchedulingEnv, self).__init__()

        self.waiting_capacity = 10000  # Maximum waiting tasks
        self.N_total_nodes = N_total_nodes
        self.total_tasks = deepcopy(total_tasks)

        self.upcoming_tasks = deepcopy(total_tasks)
        self.waiting_tasks = []
        self.executing_tasks = []
        self.available_nodes = self.N_total_nodes
        self.current_time = self.total_tasks[0]['time']

        self.average_waiting_time = 0
        self.num_executed_tasks = 0

        self.observation_space = spaces.Box(
                low=0, high=np.inf, shape=(1 + self.waiting_capacity,), dtype=np.int32
            )
        self.action_space = spaces.Discrete(self.N_total_nodes + 1)
        self.state = np.array([self.available_nodes] + [0] * self.waiting_capacity, dtype=np.int32)

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)

        self.upcoming_tasks = deepcopy(self.total_tasks)
        self.waiting_tasks = []
        self.executing_tasks = []
        self.available_nodes = self.N_total_nodes
        self.current_time = self.total_tasks[0]['time']

        padded_waiting_times = [0] * self.waiting_capacity
        state = np.array([self.available_nodes] + padded_waiting_times, dtype=np.int32)

        return state, {}

    def step(self, action):
        # Validate the action
        if action < 0 or action > self.available_nodes:
            return self.state, -1000, False, False, {}

        # Add new task to waiting list
        if self.upcoming_tasks and self.current_time == self.upcoming_tasks[0]['time']:
            self.upcoming_tasks[0]['waiting_time'] = 0
            self.waiting_tasks.append(self.upcoming_tasks.pop(0))

        # Update executing tasks
        for task in self.executing_tasks:
            task['complexity'] -= task['nodes_alloc']
            if task['complexity'] <= 0:
                self.available_nodes += task['nodes_alloc']
                self.average_waiting_time = (self.average_waiting_time*self.num_executed_tasks + task['waiting_time'])/(self.num_executed_tasks+1)
                self.num_executed_tasks += 1

        self.executing_tasks = [task for task in self.executing_tasks if task['complexity'] > 0]

        # Allocate nodes to the first waiting task
        if self.waiting_tasks and action > 0:
            self.waiting_tasks[0]['nodes_alloc'] = int(action)
            self.executing_tasks.append(self.waiting_tasks.pop(0))
            self.available_nodes -= action

        # Update waiting times
        for task in self.waiting_tasks:
            task['waiting_time'] += 1

        # Compute next state
        waiting_times = [task['waiting_time'] for task in self.waiting_tasks]
        padded_waiting_times = waiting_times + [0] * (self.waiting_capacity - len(waiting_times))

        state = np.array([self.available_nodes] + padded_waiting_times, dtype=np.int32)
        reward = -np.sum(waiting_times)
        done = len(self.upcoming_tasks) == 0 and len(self.waiting_tasks) == 0

        # print(f"Time: {self.current_time}. Allocated {action} nodes. Waiting tasks: {self.waiting_tasks}")

        self.current_time += 1
        return state, reward, done, False, {}

    def get_action_mask(self):
        # Generate a binary mask for valid actions
        mask = np.zeros(self.N_total_nodes + 1, dtype=np.int32)
        if self.available_nodes == 0:
            mask[0] = 1
        else:
          for i in range(1, self.available_nodes + 1):
              mask[i] = 1
        return mask
    
    def get_average_waiting_times(self):
        remaining_waiting_time = np.sum([np.ceil(task['complexity']/task['nodes_alloc']) for task in self.executing_tasks])
        return (self.average_waiting_time*self.num_executed_tasks + remaining_waiting_time)/(self.num_executed_tasks + len(self.executing_tasks))