import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import devo.DE
import devo.SHADE
import devo.JADE
import devo.jDE
import devo.LSHADE
import devo.CoDE
import numpy as np
import ctypes as c
import math
import argparse
from network import Net

DEFAULT_MODEL_OUT_FILE = "data/de-nn-model.pt"
DEFAULT_GYM_ENV = "CartPole-v0"
DEFAULT_CROSS_OVER_RATE = 0.7
DEFAULT_SCALING_FACTOR = 1e-4
DEFAULT_POPULATION_SIZE = 30
DEFAULT_BATCH_SIZE = 40
DEFAULT_HIDDEN_LAYER_SIZE = 40
DEFAULT_LOG_LEVEL = 1
SUPPORTED_DE_OPTIMIZERS = ['DE', 'jDE', 'SHADE', 'LSHADE', 'JADE', 'CoDE']


class Agent():
    def __init__(self, env, steps_per_episode, should_maximize):
        self.reward_inversion = should_maximize
        self.min_reward = float('inf')
        self.min_weights = None
        self.steps_per_episode = steps_per_episode
        # time step
        self.t = 0
        # OpenAI Gym env
        self.env = env
        self.model = None
        self.memory = None
        self.log_level = 0

    def attach_model(self, model):
        self.model = model
        self.init_memory(model)

    def init_memory(self, model):
        # actions played memory

        # Init randomly
        # self.memory = np.random.uniform(-1.0, 1.0,
        #                                 (model.N, model.D_in))

        # Init zeros
        self.memory = np.zeros((model.N, model.D_in))

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
            if self.log_level > 1:
                print("Weights:", weights_vec)
                print("Warning: nan encountered.  Ignoring individual.")
            return c.c_double(0.0)

        model.backward(weights_vec)

        # run with new values
        reward = -self.run_episode(self.steps_per_episode)
        if self.t % 100 == 0:
            if self.log_level > 0:
                print("{}: reward: {}".format(self.t, reward))

        if reward < self.min_reward:
            self.min_reward = reward
            self.min_weights = weights_vec

        return c.c_double(reward)  # return fitness values

    def run_episode(self, steps, should_render=False):
        model = self.model
        env = self.env
        # generate next batch
        observation = env.reset()

        # load from memory
        observations = self.memory + observation

        # add observation
        observations = np.concatenate(
            (observations[1:, ], np.expand_dims(observation, axis=0)))

        rewards = []
        episode_duration = 0
        for t in range(steps):
            episode_duration += 1
            if should_render or t % 20 == 0:
                env.render()
            x_t = torch.from_numpy(observations).float()
            y = model(x_t)  # get action list
            action_f = y[-1].item()  # get last action in list
            if math.isnan(action_f):
                if self.log_level > 1:
                    print("Warning: action is nan -- ", y)
                    print(" x_t --- ", x_t)
                return 0.0
            action = round(action_f)
            observation, reward, done, info = env.step(action)

            # add new observation
            observations = np.concatenate(
                (observations[1:, ], np.expand_dims(observation, axis=0)))

            rewards.append(reward / (t + 1))  # discount reward over time
            if done:
                if should_render:
                    print("Episode finished after {} steps. reward: {}, info: {}".format(
                        t+1, reward, info))
                break

        reward_sum = sum(rewards)
        avg_reward = reward_sum / episode_duration

        # save latest observations to memory
        self.memory = observations

        # When maximizing, rewards need to be inverted
        if self.reward_inversion:
            return -avg_reward
        else:
            return avg_reward

    # Run forever
    def run_forever(self, steps=100000):
        while True:
            result = self.run_episode(steps, True)
            print("Reward result:", result)

    def results_callback(self, population, fitness_values, population_size, problem_size):
        print("Completed training.")
        return None


def run(args):
    env = gym.make(args.env_name)
    observation = env.reset()
    steps_per_episode = args.steps
    model_file_output_path = args.from_file

    # N is episode steps length; D_in is input observation dimension;
    # H is hidden layer dimension; D_out is output action space dimension.
    N, D_in, H, D_out = args.batch_size, observation.shape[0], 40, 1

    agent = Agent(env, steps_per_episode, args.maximize)
    model = Net(N, D_in, H, D_out, agent)

    # set log level
    agent.log_level = args.log_level

    if args.load_model:
        model.load_state_dict(torch.load(model_file_output_path))
        model.eval()

    # connect network to agent
    agent.attach_model(model)

    if args.load_model:
        agent.run_forever(steps_per_episode)
        exit()

    # fill
    model.train()

    scaling_factor = args.scaling_factor
    crossover_rate = args.crossover_rate
    population_size = args.population_size
    episodes_num = args.episodes  # number of episodes

    # generate flattened weights
    model.flatten()

    problem_size = model.flattened.shape[0]

    print("problem_size: ", problem_size)

    # Initial population, Fitness values
    x = torch.randn(population_size, problem_size)
    y = torch.randn(population_size, D_out)

    # Convert to c pointers
    x_c = x.detach().numpy().ctypes.data_as(
        c.POINTER(c.c_double))  # c pointer init population
    y_c = y.detach().numpy().ctypes.data_as(
        c.POINTER(c.c_double))  # c pointer init fitness values

    # TODO: make these adjustable
    optimizer = getattr(devo, args.optimizer_name)

    # # Using Adaptive-DEs
    optimizer.run(
        episodes_num,
        population_size,  # population size
        scaling_factor,  # scaling factor
        crossover_rate,  # crossover rate
        agent.objective_func,
        problem_size,  # problem size
        -100,  # unused value
        100,  # unused value
        x_c,
        y_c,
        agent.results_callback  # no results callback needed
    )

    # Get mins - inverted in output
    print("min_fitness: ", agent.min_reward)

    model.update_weights_from_vec(agent.min_weights)

    result = -agent.run_episode(agent.steps_per_episode, True)

    if args.should_test:
        print("test_run(expected: {}, actual: {})".format(
            agent.min_reward, result))

    env.close()

    if args.save_model:
        print("model_file: ", model_file_output_path)
        # save model
        torch.save(model.state_dict(), model_file_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Adaptive-DEs Example')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--steps', type=int, default=300,
                        help='Steps per episode (default: 300)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes per objective functtion run (default: 1000)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Load from saved model')
    parser.add_argument('--optimizer-name', type=str, default='DE',
                        help='Choose a devo Adaptive-DE optimizer ({})'.format(SUPPORTED_DE_OPTIMIZERS))
    parser.add_argument('--log-level', type=int, default=DEFAULT_LOG_LEVEL,
                        help='Set log level (default: {})'.format(DEFAULT_LOG_LEVEL))
    parser.add_argument('--maximize', action='store_true', default=False,
                        help='Maximize rather than minimize (default: {})'.format(False))
    parser.add_argument('--should-test', action='store_true', default=True,
                        help='Should run a test at the end of training (default: {})'.format(True))
    parser.add_argument('--env-name', type=str, default=DEFAULT_GYM_ENV,
                        help='OpenAI Gym environment to use (default: {})'.format(DEFAULT_GYM_ENV))
    parser.add_argument('--from-file', type=str, default=DEFAULT_MODEL_OUT_FILE,
                        help='Choose file name for loading/saving (default: {})'.format(DEFAULT_MODEL_OUT_FILE))
    parser.add_argument('--crossover-rate', type=float, default=DEFAULT_CROSS_OVER_RATE,
                        help='Choose crossover rate for devo optimizer (default: {})'.format(DEFAULT_CROSS_OVER_RATE))
    parser.add_argument('--scaling-factor', type=float, default=DEFAULT_SCALING_FACTOR,
                        help='Choose scaling factor for devo optimizer (default: {})'.format(DEFAULT_SCALING_FACTOR))
    parser.add_argument('--population-size', type=float, default=DEFAULT_POPULATION_SIZE,
                        help='Choose population size for devo optimizer (default: {})'.format(DEFAULT_POPULATION_SIZE))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Choose batch size of actions during episode (default: {})'.format(
                            DEFAULT_BATCH_SIZE)
                        )

    parser.add_argument('--hidden-layer-size', type=int, default=DEFAULT_HIDDEN_LAYER_SIZE,
                        help='Choose hidden layer size of actions during episode (default: {})'.format(
                            DEFAULT_HIDDEN_LAYER_SIZE)
                        )

    args = parser.parse_args()

    run(args)