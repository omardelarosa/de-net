import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import devo.DE as DE
import numpy as np
import ctypes as c
import math
from network import Net


class Agent():
    def __init__(self, env, steps_per_episode):
        self.min_reward = float('inf')
        self.min_weights = None
        self.steps_per_episode = steps_per_episode
        # time step
        self.t = 0
        # OpenAI Gym env
        self.env = env
        self.model = None

    def attach_model(self, model):
        self.model = model

    def init_memory(self, model):
        # actions played memory
        self.memory = np.random.rand(
            model.N, model.D_in).tolist()

    # DE stuff
    def objective_func(self, vec, dimension):
        model = self.model
        self.t = self.t + 1
        weights_vec = np.zeros((dimension,), dtype='float32')
        # gather data
        for i in range(dimension):
            v = np.float32(vec[i])
            if not math.isnan(v):
                weights_vec[i] = v
            else:
                weights_vec[i] = 0.0

        nans = weights_vec[np.isnan(weights_vec)]
        # print("nans", nans)
        if len(nans) > 0:
            print(weights_vec)
            print("Warning: nan encountered.  Ignoring individual.")
            return c.c_double(0.0)
        model.update_weights_from_vec(weights_vec)

        # run with new values
        reward = -self.run_episode(self.steps_per_episode)
        if self.t % 100 == 0:
            print("{}: reward: {}".format(self.t, reward))

        if reward < self.min_reward:
            self.min_reward = reward
            self.min_weights = weights_vec

        # if abs(reward) == float('inf'):
        #     print("Warning: divergent reward.")
        #     exit(1)
        return c.c_double(reward)  # return fitness values

    def run_episode(self, steps, should_render=False):
        model = self.model
        env = self.env
        # generate next batch
        observation = env.reset()
        reward_sum = 0.0
        for t in range(steps):
            if should_render or t % 100 == 0:
                env.render()
            x_t = torch.from_numpy(observation).float().unsqueeze(0)
            y = model(x_t)  # get action list
            action_f = y[-1].item()  # get last action in list
            if math.isnan(action_f):
                print("Warning: action is nan -- ", y)
                print(" x_t --- ", x_t)
                return 0.0
            action = round(action_f)
            observation, reward, done, info = env.step(action)
            reward_sum += reward  # discount reward over time
            if done:
                break
        avg_reward = reward_sum / steps
        return avg_reward

    def run(self, max_steps=20000):
        s = 0
        model = self.model
        env = self.env
        obs = env.reset()
        while s < max_steps:
            x_t = torch.from_numpy(obs).float().unsqueeze(0)
            y = model(x_t)
            action_f = y[0].item()
            if math.isnan(action_f):
                print("Warning: action is nan -- ", y)
                print(" x_t --- ", x_t)
                return 0.0
            action = round(action_f)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
            s = s + 1
        return

    def results_callback(self, population, fitness_values, population_size, problem_size):
        print("Completed training.")
        return None


def train():
    env = gym.make('CartPole-v0')
    observation = env.reset()
    steps_per_episode = 300
    # N is episode steps length; D_in is input observation dimension;
    # H is hidden layer dimension; D_out is output action space dimension.
    N, D_in, H, D_out = 1, observation.shape[0], 40, 1

    agent = Agent(env, steps_per_episode)
    model = Net(N, D_in, H, D_out, agent)

    # connect network to agent
    agent.attach_model(model)
    agent.init_memory(model)

    # fill
    model.train()

    scaling_factor = 1e-4
    crossover_rate = 5e-1
    population_size = 30
    steps = 5000

    # generate flattened weights
    model.flatten()

    problem_size = model.flattened.shape[0]

    print("Parameters: ", problem_size)

    # Initial population, Fitness values
    x = torch.randn(population_size, problem_size)
    y = torch.randn(population_size, D_out)

    # Convert to c pointers
    x_c = x.detach().numpy().ctypes.data_as(
        c.POINTER(c.c_double))  # c pointer init population
    y_c = y.detach().numpy().ctypes.data_as(
        c.POINTER(c.c_double))  # c pointer init fitness values

    # # Using Adaptive-DEs
    DE.run(
        steps,
        population_size,  # population size
        scaling_factor,  # scaling factor
        crossover_rate,  # crossover rate
        agent.objective_func,
        problem_size,  # problem size
        -100,
        100,
        x_c,
        y_c,
        agent.results_callback  # no results callback needed
    )

    # Get mins
    print("Min Reward", agent.min_reward)
    print("Min weights", agent.min_weights)

    model.update_weights_from_vec(agent.min_weights)

    result = -agent.run_episode(agent.steps_per_episode, True)

    print("Results:  Expected {}, Actual {}".format(
        agent.min_reward, result))

    agent.run_episode(20000, True)


if __name__ == '__main__':
    train()