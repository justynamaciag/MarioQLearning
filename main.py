import random

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from collections import defaultdict

moves = [0, 1, 2, 3, 4, 5, 6]


class MarioQLearner:
    def __init__(self, env, alpha, gamma, epsilon):
        self.quality = defaultdict(lambda: 0)
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def discretize(self, state):
        print(state)
        print("state: " , state.tobytes())
        return state
        # return get_bucket(0), get_bucket(1), get_bucket(2), get_bucket(3)

    def update_knowledge(self, action, state, new_state, reward):
        actions = [self.quality[(new_state.tobytes(), move)] for move in moves]
        print(actions)
        max_factor = max(actions)
        q = (1 - self.alpha) * self.quality[(state, action)] + self.alpha * (reward + self.gamma * max_factor)
        print(q)
        self.quality[(state, action)] = q

    def pick_action(self, state):
        def get_random_action():
            return 1

        if random.random() < self.epsilon:
            return get_random_action()
        else:
            actions = {move: self.quality[(state.tobytes(), move)] for move in moves}
            actions = {key: val for key, val in sorted(actions.items(), key=lambda item: item[1])}

            first_move, first_val = next(iter(actions.items()))
            second_move, second_val = next(iter(actions.items()))

            if first_val == second_val:
                return get_random_action()
            else:
                return first_move

    def action(self):
        done = False
        for step in range(5000):
            reward_sum = 0.0
            state = self.env.reset()
            while not done:
                action = self.pick_action(state)
                #calling mario
                new_state, reward, done, info = self.env.step(action)
                print("info :", info)
                print("reward:", reward)
                new_state = self.discretize(new_state)
                self.update_knowledge(action, state.tobytes(), new_state, reward)
                state = new_state
                reward_sum += reward
            self.env.render()


            print(reward_sum)


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    alpha, gamma, epsilon = 0.8, 1, 0.3
    marioQLearner = MarioQLearner(env, alpha, gamma, epsilon)
    marioQLearner.action()
    env.close()


if __name__ == "__main__":
    main()
