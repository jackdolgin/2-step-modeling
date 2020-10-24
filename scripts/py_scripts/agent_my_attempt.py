import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from utils import softmax
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random
import itertools

pd.options.display.float_format = '{:.2f}'.format



class Context:
    def __init__(self, transitions_to_arrive, stage, choices, n_choices, n_steps):
        self.stage = stage
        self.earlier_states = transitions_to_arrive[:-1]
        self.current_state = transitions_to_arrive[-1:]
        self.rand_acts = random.sample(choices, n_choices)
        self.state_pairs = {i: j for i, j in zip(choices,
                                                 self.rand_acts)}
        if self.stage == n_steps:
            self.likely_prob = .75
            self.r = random.choice(range(2))# change this line
        else:
            self.likely_prob = .7
            self.r = 0
        self.unlikely_prob = (1 - self.likely_prob) / (n_choices - 1)
        self.Q = {i: np.random.uniform(0, 1e-4) for i in choices}

class Agent:
    def __init__(self, steps, n_steps, choices, n_choices):
        self.steps = steps
        self.n_steps = n_steps
        self.choices = choices
        self.n_choices = n_choices
        self.all_contexts = {}
        self.create_contexts()

    def run(self):
        states_passed = [self.all_contexts]
        picks = []

        for step in range(self.n_steps):
            states_passed_t = tuple(states_passed)
            choice = self.choose_choice(states_passed_t,
                                        step)
            picks.append(choice)

            state = self.transition(choice)
            states_passed.append(state)

        self.reward(states_passed, picks)

    def create_contexts(self):
        for i in range(self.n_steps + 1):
            choice_combos = list(
                itertools.product(self.choices, repeat=i))
            for x in choice_combos:
                choice_combo = Context(x, i,
                                       self.choices,
                                       self.n_choices,
                                       self.n_steps)
                self.all_contexts[x] = choice_combo
        return self.all_contexts

    def choose_choice(self, context, step):
        p = softmax(self.all_contexts[context].Q,
                    self.steps[step][1])
        choice = np.random.choice(self.choices, p=p)
        return choice

    def transition(self, choice):
        potential_new_contexts = {}
        prev_context = self.all_contexts[states_passed_t]
        likeliest_state= prev_context.state_pairs[choice]
        for i in self.all_contexts.values():
            if i.earlier_states == states_passed_t:
                if i.current_state[0] == likeliest_state:
                    p = i.likely_prob
                else:
                    p = i.unlikely_prob
                potential_new_contexts[i]: p

        prob_total = 0
        cutoff = np.random.rand()
        for possible_state, prob in potential_new_contexts.items():
            prob_total += prob
            if prob_total >= cutoff:
                return possible_state

    def reward(self, states_passed, picks):
        delta_sum = 0
        # q_1 = 0
        for i, j in enumerate(list(reversed(picks))):
            imm_reward = self.get_context_val(0, 'r')
            q_0 = self.extract_reward(1, 'Q')
            delta_sum = 0
            for a, b in enumerate(j, ):
                j.Q += (lambda_var ** i) *
            delta_0 = imm_reward + q_1 - q_0
            q_1 = j.Q

            j.qvalue += self.alpha * (i * self.lambda * j.delta + (i + 1) * picks[i + 1].qvalue)

        j.value += delta_sum

    def get_context_val(self, x, y):
        given_context = states_passed[:n_steps - i - x]
        given_context = tuple(given_context)
        context_val = self.all_contexts[given_context]
        context_attr = getattr(context_val, y)
        return context_val

        imm_reward = all_contexts[]

def run_single_softmax_experiment(self, choices, steps, trials):
    """Run experiment with agent using softmax update rule."""
    print('Running a contextual bandit experiment')

    n_steps = len(steps)
    n_choices = len(choices)

    agent = Agent(steps, n_steps, choices, n_choices)

    for _ in range(trials):
        agent.run()
    df = DataFrame(agent.log, columns=ca.choices_et_al)
    table = pa.Table.from_pandas(df)
    fn = os.path.join('..', '..', 'data', 'softmax_experiment.parquet')
    pq.write_table(table, fn)

    globals().update(locals())  #

    return df

if __name__ == '__main__':
    np.random.seed(42)
    choices = ("left", "right")
    steps = ((.1, .5), (.1, .5)) # implicitly sets how many steps the task is based on this length; the first value in each tuple is the alpha, and the second is the beta
    # print('Running experiment with alpha={} and beta={}'.format(alpha, beta))
    trials = 360
    run_single_softmax_experiment(choices, steps, trials)
    # import vis
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # vis.plot_simulation_run()