import gym
from gym import spaces
import json
import random

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

        # Load existing data from the file and populate used_coords
        with open("map.json", 'r') as fstream:
            existing_data = json.load(fstream)
        
        for i in existing_data:
            for x in i["tiles"]:
                self.used_coords.append((x["x"], x["y"]))
            for j in i["objects"]:
                self.used_coords.append((j["x"], j["y"]))

    def reset(self, population, city_type):
        # Reset the environment to a new state
        self.population = population
        self.city_type = city_type

        return self._get_observation()

    def step(self, action):
        # Perform the given action and return the resulting state, reward, and done flag
        if action == 0:  # Add House
            pass

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
        pass

    def _is_done(self):
        pass

    def render(self):
        pass

    def save_map(self, filename):
        pass

    def load_map(self, filename):
        pass


city_env = CityBuilderEnv()
print(city_env.used_coords)
