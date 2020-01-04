# DENet - Differential Evolution + Neural Networks

This is a neural network designed to use Differential Evolution rather than gradient descent for updating its weights.

## Installation

```bash
pipenv install
pipenv shell
```

## Usage

```
python de-net.py -husage: de-net.py [-h] [--save-model] [--steps STEPS] [--episodes EPISODES]
                 [--load-model] [--optimizer-name OPTIMIZER_NAME]
                 [--log-level LOG_LEVEL] [--maximize] [--should-test]
                 [--env-name ENV_NAME] [--from-file FROM_FILE]
                 [--crossover-rate CROSSOVER_RATE]
                 [--scaling-factor SCALING_FACTOR]
                 [--population-size POPULATION_SIZE] [--batch-size BATCH_SIZE]
                 [--hidden-layer-size HIDDEN_LAYER_SIZE]

PyTorch Adaptive-DEs Example

optional arguments:
  -h, --help            show this help message and exit
  --save-model          For Saving the current Model
  --steps STEPS         Steps per episode (default: 300)
  --episodes EPISODES   Number of episodes per objective functtion run (default: 1000)
  --load-model          Load from saved model
  --optimizer-name OPTIMIZER_NAME
                        Choose a devo Adaptive-DE optimizer (['DE', 'jDE',
                        'SHADE', 'LSHADE', 'JADE', 'CoDE'])
  --log-level LOG_LEVEL
                        Set log level (default: 1)
  --maximize            Maximize rather than minimize (default: False)
  --should-test         Should run a test at the end of training (default:
                        True)
  --env-name ENV_NAME   OpenAI Gym environment to use (default: CartPole-v0)
  --from-file FROM_FILE
                        Choose file name for loading/saving (default: data/de-
                        nn-model.pt)
  --crossover-rate CROSSOVER_RATE
                        Choose crossover rate for devo optimizer (default:
                        0.7)
  --scaling-factor SCALING_FACTOR
                        Choose scaling factor for devo optimizer (default:
                        0.0001)
  --population-size POPULATION_SIZE
                        Choose population size for devo optimizer (default:
                        30)
  --batch-size BATCH_SIZE
                        Choose batch size of actions during episode (default:
                        40)
  --hidden-layer-size HIDDEN_LAYER_SIZE
                        Choose hidden layer size of actions during episode
                        (default: 40)
```
