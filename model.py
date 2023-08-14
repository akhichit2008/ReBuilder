import gym
from gym import spaces
from agents import *
import json
import random
import os
import numpy as np

class CityBuilderEnv(gym.Env):
    def __init__(self):
        super(CityBuilderEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(3)  # Three actions: 0 - Add House, 1 - Add Factory, 2 - Add Garden
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=float)  # Population, City Type

        # Set the initial state
        self.population = None
        self.city_type = None
        self.used_coords = []
        self.data = {}
        self.map_data = {}

        # Load existing data from the file and populate used_coords
        self.fstream = open("model_testing.json","r")
        self.map_data = existing_data = json.load(self.fstream)
        for i in existing_data["objects"]:
            self.used_coords.append((i["x"], i["y"]))
    def reset(self, population, city_type):
        # Reset the environment to a new state
        self.population = population
        self.city_type = city_type

        return self._get_observation()

    def step(self, action):
    # ... (Previous implementation remains the same) ...

        if action == 0:  # Add House
            f = True
            while f:
                x = random.randint(1, 400)
                y = random.randint(1, 400)
                if (x, y) not in [(obj["x"], obj["y"]) for obj in self.map_data["objects"]]:
                    new_object = {
                    "x": x,
                    "y": y,
                    "type": "house"
                    }
                    self.map_data["objects"].append(new_object)
                    f = False

    # ... (Action 1 and 2 remain unchanged) ...

    # Write the updated JSON data back to the file, appending only the new entry
                    with open("map.json", "w") as file:
                            json.dump(self.map_data, file, indent=4)

    # ... (Reward calculation and termination checks remain unchanged) ...

                    
                else:
                    continue


        elif action == 1:  # Add Factory
            pass

        elif action == 2:  # Add Garden
            pass

        else:
            raise ValueError("Invalid action.")

        # Calculate the reward based on the current state
        reward = self._calculate_reward()

        # Check if the goal is achieved or the maximum number of steps is reached
        done = self._is_done()

        # Return the resulting state, reward, and done flag
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Return the current state as the observation
        return [self.population, self.city_type]

    def _calculate_reward(self):
    # Get the current population and city type from the observation
        current_population, current_city_type = self._get_observation()

    # Calculate the reward based on the changes in population and building additions
        reward = 0

    # Calculate the population increment based on the difference from the previous population
        if self.population is not None:
            population_increment = self.population - current_population
            reward += population_increment

    # Calculate the reward for each building type based on the number of buildings added
        num_houses_added = 0
        num_factories_added = 0
        num_gardens_added = 0

        for obj in self.map_data["objects"]:
            if obj["type"] == "house":
                num_houses_added += 1
            elif obj["type"] == "factory":
                num_factories_added += 1
            elif obj["type"] == "garden":
                num_gardens_added += 1

        reward += num_houses_added
        reward += 3 * num_factories_added
        reward += 2 * num_gardens_added

        return reward


    def _is_done(self):
        pass

    def render(self):
        os.system("python main.py")

    def save_map(self, filename):
        pass

    def load_map(self, filename):
        pass


city_env = CityBuilderEnv()
'''
print(city_env.used_coords)
city_env.step(0)
city_env.render()
city_env.step(1)
city_env.step(2)
'''
# Define Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Create the environment (CityBuilderEnv instance)
city_env = CityBuilderEnv()

# Create the Q-learning agent
q_agent = QLearningAgent(city_env.action_space, city_env.observation_space)

# Q-learning training loop
for episode in range(num_episodes):
    state = city_env.reset(population=np.random.uniform(0, 1), city_type=np.random.uniform(0, 1))
    done = False

    while not done:
        # Choose action using epsilon-greedy strategy
        action = q_agent.choose_action(state)
        
        next_state, reward, done, _ = city_env.step(action)
        next_state_idx = np.argmax(next_state)
        
        # Update Q-values
        q_agent.update_q_table(state, action, reward, next_state_idx)

        state = next_state

print("Training completed.")

# Evaluate the trained agent
num_eval_episodes = 10
total_rewards = 0

for _ in range(num_eval_episodes):
    state = city_env.reset(population=np.random.uniform(0, 1), city_type=np.random.uniform(0, 1))
    done = False

    while not done:
        action = np.argmax(q_agent.q_table[state, :])
        next_state, reward, done, _ = city_env.step(action)
        total_rewards += reward
        state = next_state

average_rewards = total_rewards / num_eval_episodes
print("Average rewards:", average_rewards)
