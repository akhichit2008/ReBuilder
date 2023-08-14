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
             f = True
            while f:
                x = random.randint(1, 400)
                y = random.randint(1, 400)
                if (x, y) not in [(obj["x"], obj["y"]) for obj in self.map_data["objects"]]:
                    new_object = {
                    "x": x,
                    "y": y,
                    "type": "factory"
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

        elif action == 2:  # Add Garden
             f = True
            while f:
                x = random.randint(1, 400)
                y = random.randint(1, 400)
                if (x, y) not in [(obj["x"], obj["y"]) for obj in self.map_data["objects"]]:
                    new_object = {
                    "x": x,
                    "y": y,
                    "type": "garden"
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
        pass

    def save_map(self, filename):
        pass

    def load_map(self, filename):
        pass


city_env = CityBuilderEnv()
print(city_env.used_coords)
city_env.step(0)
