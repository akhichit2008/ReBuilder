import numpy as np

class QLearningAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        self.num_bins = 10  # Number of bins for each dimension
        self.state_space_size = self.num_bins ** len(observation_space.low)
        self.q_table = np.zeros((self.state_space_size, action_space.n))
        
        self.discrete_ranges = [
            np.linspace(low, high, self.num_bins)
            for low, high in zip(observation_space.low, observation_space.high)
        ]
    
    def discretize_value(self, value, range_array):
        if value <= range_array[0]:
            return 0
        elif value >= range_array[-1]:
            return self.num_bins - 1
        else:
            return np.digitize(value, range_array) - 1
    
    def discretize_observation(self, observation):
        discrete_indices = [
            self.discretize_value(value, range_array)
            for value, range_array in zip(observation, self.discrete_ranges)
        ]
        state_idx = np.ravel_multi_index(discrete_indices, [self.num_bins] * len(observation))
        return state_idx
    
    def choose_action(self, observation):
        state = self.discretize_observation(observation)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation
    
    def update_q_table(self, state, action, reward, next_state):
        state_idx = self.discretize_observation(state)
        next_state_idx = self.discretize_observation(next_state)
        
        self.q_table[state_idx, action] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state_idx, :]) - self.q_table[state_idx, action]
        )
