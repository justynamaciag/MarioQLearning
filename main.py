import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from collections import defaultdict
from statistics_generator import StatsGenerator
import json

moves = [1, 2, 3, 4, 5]


class MarioQLearner:
    def __init__(self, env, alpha, gamma, epsilon, stats_gen, dict_filepath=None):
        self.quality = defaultdict(lambda: 0)
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.filename = dict_filepath
        self.env_counter = 1
        self.stats_gen = stats_gen
        self.current_lifes = 0

    def discretize(self, state):
        return state[0] - state[0] % 10, state[1] - state[1] % 10

    def update_knowledge(self, prev_action, prev_state, action, state, reward):
        old_value = self.quality[(prev_state, prev_action)]
        next_val = self.quality[(state, action)]
        learned = self.alpha * (reward + self.gamma * next_val - old_value)
        update = old_value + learned
        self.quality[(prev_state, prev_action)] = update

    def pick_action(self, state):
        def get_random_action():
            return random.choice(moves)

        if random.random() < self.epsilon:
            self.epsilon -= 0.0001

            return get_random_action()
        else:
            actions = {move: self.quality[(state, move)] for move in moves}
            actions = {key: val for key, val in sorted(actions.items(), key=lambda item: item[1])}

            first_move, first_val = next(iter(actions.items()))
            second_move, second_val = next(iter(actions.items()))

            if first_val == second_val:
                return get_random_action()
            else:
                return first_val

    def read_dictionary(self):
        if self.filename:
            with open(self.filename, 'r') as f:
                try:
                    data = json.load(f)
                    for i in data.items():
                        key = i[0].split(",")
                        observation = int(key[0]), int(key[1])
                        self.quality[(observation), int(key[2])] = int(i[1])
                except ValueError:
                    self.quality = defaultdict(lambda: 0)
        else:
            self.quality = defaultdict(lambda: 0)

    def action(self):
            self.read_dictionary()
            for step in range(100):
                self.env.reset()
                prev_state = self.discretize((40, 79))
                prev_action = self.pick_action(prev_state)
                _, reward, done, info = self.env.step(prev_action)
                state = (info['x_pos'], info['y_pos'])

                self.current_lifes = info['life']

                reward_sum = reward

                while not done:
                    state = self.discretize(state)

                    action = self.pick_action(state)

                    self.update_knowledge(prev_action, prev_state, action, state, reward)

                    prev_state = state
                    _, reward, done, info = self.env.step(action)

                    state = info['x_pos'], info['y_pos']

                    prev_action = action

                    reward_sum += reward
                    self.env.render()

                    if self.current_lifes != info['life']:
                        self.current_lifes = info['life']
                        self.stats_gen.save_stats(self.env_counter, info['life'], reward_sum)
                        reward_sum = 0.0

                self.env_counter += 1

                with open(self.filename, 'w') as f:
                    data = {}
                    for key, val in self.quality.items():
                        data[",".join([str(key[0][0]), str(key[0][1]), str(key[1])])] = val

                    json.dump(data, f)


def main():
    stats_gen = StatsGenerator(1, 'results/final_tats.txt')

    env = gym_super_mario_bros.make('SuperMarioBros-v0')

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    alpha, gamma, epsilon = 0.1, 1, 0.3
    marioQLearner = MarioQLearner(env, alpha, gamma, epsilon, stats_gen)
    marioQLearner.action()
    env.close()


if __name__ == "__main__":
    main()
