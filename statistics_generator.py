import matplotlib.pyplot as plt


class StatsGenerator:
    def __init__(self, run_iteration, filepath):
        self.run_iteration = run_iteration
        self.filepath = filepath

    def save_stats(self, env_iteration, life, reward):
        with open(self.filepath, 'a+') as f:
            f.write(';'.join([str(self.run_iteration), str(env_iteration), str(life), str(reward)]) + "\n")


    def create_diagram(self):
        rewards = []
        with open(self.filepath, 'r') as f:
            for i in f.readlines():
                line = i.split(";")
                rewards.append(float(line[3].strip()))
        print(rewards)

        plt.plot([i for i in range(len(rewards))], rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')

        plt.show()
