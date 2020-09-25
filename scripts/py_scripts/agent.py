import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from utils import softmax
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random

pd.options.display.float_format = '{:.2f}'.format


class ContextualBandit(object):
    def __init__(self):
        self.actions = (23, 14, 8, 3)
        self.n_actions = len(self.actions)
        # Contexts and their probabilities of winning
        self.contexts = dict(SLL=0.2, SLR=0.5, SRL=0.8, SRR=.4)
        self.get_context()
        self.actions_in_text = tuple(map(
            lambda x: 'Q(c,' + str(x) + ')',
            self.actions))
    def get_context_list(self):
        return list(self.contexts.keys())

    def get_context(self):
        self.context = np.random.choice(list(self.contexts.keys()))
        return self.context

    def reward(self, action):
        if action not in self.actions:
            print('Error: action not in', self.actions)
            sys.exit(-1)
        p = self.contexts[self.context]
        if np.random.rand() < p:
            r = action
        else:
            r = -action
        return r


class ContextualAgent(object):
    def __init__(self, bandit, beta=None, alpha=None):
        self.beta = beta
        self.bandit = bandit
        self.contexts = self.bandit.get_context_list()
        self.actions = self.bandit.actions
        self.actions_in_text = self.bandit.actions_in_text
        self.n_actions = bandit.n_actions
        self.et_al = ('context', 'reward', 'action')
        self.actions_et_al = self.actions_in_text + self.et_al
        self.alpha = alpha
        self.Q = {}

        # init with small random numbers to avoid ties
        for context in self.contexts:
            self.Q[context] = np.random.uniform(0, 1e-4, self.n_actions)

        self.log = {k: [] for k in self.actions_et_al}

    def run(self):

        context = self.bandit.get_context()
        action = self.choose_action(context)
        reward = self.bandit.reward(self.actions[action])


        # Update action-value
        self.update_action_value(context, reward, action)

        # Keep track of performance
        for i, j in enumerate(self.actions_et_al):
            if i < self.n_actions:
                self.log[j].append(self.Q[context][i])
            elif i < self.n_actions + 2:
                self.log[j].append(eval(j))
            else:
                self.log['action'].append(self.actions[action])

    def choose_action(self, context):
        p = softmax(self.Q[context], self.beta)
        actions = range(self.n_actions)
        action = np.random.choice(actions, p=p)
        return action

    def update_action_value(self, context, reward, action):  # going to have to add a lambda for updating the action1 value
        error = reward - self.Q[context][action]
        self.Q[context][action] += self.alpha * error

def run_single_softmax_experiment(beta, alpha):
    """Run experiment with agent using softmax update rule."""
    print('Running a contextual bandit experiment')
    cb = ContextualBandit()
    ca = ContextualAgent(cb, beta=beta, alpha=alpha)
    trials = 360

    for _ in range(trials):
        ca.run()
    df = DataFrame(ca.log, columns=ca.actions_et_al)
    table = pa.Table.from_pandas(df)
    fn = os.path.join('..', '..', 'data', 'softmax_experiment.parquet')
    pq.write_table(table, fn)

    globals().update(locals())  #

    return df


if __name__ == '__main__':
    np.random.seed(42)
    beta = 0.5
    alpha = 0.1
    print('Running experiment with alpha={} and beta={}'.format(alpha, beta))
    run_single_softmax_experiment(beta, alpha)
    # import vis
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # vis.plot_simulation_run()

