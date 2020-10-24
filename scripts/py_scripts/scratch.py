import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from utils import softmax, normal_round
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random
import itertools
from copy import copy

pd.options.display.float_format = '{:.2f}'.format


class Sequence:
    def __init__(self, transitions_to_arrive, stage, n_choices, rwd_prob_to_inflate, added_prob, n_rwds, a_mult):
        self.stage = stage
        self.earlier_states = transitions_to_arrive[:-1]
        self.current_state = transitions_to_arrive[-1:]
        self.rand_acts = random.sample(choices, n_choices)
        self.action_pairs = {i: j for i, j in zip(choices,
                                                  # self.rand_acts)}
                                                  choices)}
        self.rwd_prob_to_inflate = rwd_prob_to_inflate
        self.added_prob = added_prob
        self.n_rwds = n_rwds
        for key, val in stages[(self.stage - 1) * a_mult].items():
            setattr(self, key, val)

        self.unlikely_trans_prob = ((1 - self.likely_trans_prob)
                                    / (n_choices - 1))
        self.Q = {i: np.random.uniform(0, 1e-4) for i in choices} # can think of this `Q` more as the Q of taking the action from a previous state that attempts to reach this further-along state

        for key in self.reward_probs.keys():
            if key == self.rwd_prob_to_inflate:
                self.reward_probs[key] += self.added_prob
            else:
                self.reward_probs[key] -= self.added_prob / (self.n_rwds - 1)

    def random_walk(self):
        pass

class Agent:
    def __init__(self, steps, n_stages, n_choices, cols_for_df):
        self.steps = steps
        self.n_stages = n_stages
        self.n_choices = n_choices
        self.all_sequences = {}
        self.create_sequences()

        self.log = {}
        for i in cols_for_df:
            setattr(self, i, None)

    def run(self, trial):
        self.trial = trial
        self.past_states = []
        self.picks = []

        for step in range(self.n_stages):
            self.step = step
            self.choose_choice()

            self.transition()
            self.q_update()
            # self.transition()
            self.logger(("imm_reward", ))

        [i.random_walk() for i in self.all_sequences.values()]

    def choose_choice(self):
        p = softmax(self.get_state_val(self.past_states, 'Q'),
                    self.get_state_val(self.past_states, 'beta'))

        self.choice = np.random.choice(choices, p=p)

        self.logger(("trial", "step", "past_states", "picks", "choice"))

        self.picks.append(self.choice)


    def transition(self):
        potential_new_states = {}
        possible_action_pairs = self.get_state_val(self.past_states,
                                                   "action_pairs")
        likeliest_state = possible_action_pairs[self.choice]
        for sequence in self.all_sequences.values():
            if (sequence.earlier_states == tuple(self.past_states)
                    and sequence.stage > 0):
                possible_state = sequence.current_state[0]
                if possible_state == likeliest_state:
                    prob = sequence.likely_trans_prob
                else:
                    prob = sequence.unlikely_trans_prob
                potential_new_states[possible_state] = prob

        self.outcome = np.random.choice(list(potential_new_states.keys()),
                                        p=list(potential_new_states.values()))
        self.outcome_chance = potential_new_states[self.outcome]

        self.past_states.append(self.outcome)
        self.updated_state = self.past_states
        self.logger(("outcome", "outcome_chance", "updated_state"))

        # self.log['updated_state'].append(self.past_states)



        # log choice, past_states, outcome, past_states, past_picks, and probability of getting that outcome

        # return outcome
        # prob_total = 0
        # cutoff = np.random.rand()
        # for possible_state, prob in potential_new_states.items():
        #     prob_total += prob
        #     if prob_total >= cutoff:
        #         return possible_state

    def q_update(self):
        reward_probs = self.index_get_state_val(-1, 'reward_probs', 'y')

        self.imm_reward = np.random.choice(list(reward_probs.keys()),
                                           p=list(reward_probs.values()))

        q_1 = self.index_get_state_val(-1, 'Q', 'y')
        if self.step == 0:
            q_0 = q_1
        else:
            q_0 = self.index_get_state_val(0, 'Q', 'y')
        delta = self.imm_reward + q_1 - q_0

        # log reward and all the q's here
        # tuple(map(
        #     lambda x: 'Q(c,' + str(x) + ')',
        #     self.actions))

        for i in range(self.step):
            the_Q = self.index_get_state_val(i, 'Q', 'n')
            alpha = self.index_get_state_val(i, 'alpha', 'y')
            change = delta * alpha * (lmbda ** i)
            the_Q[0][the_Q[1]] += change  # this is where the actual update occurs

    def index_get_state_val(self, x, y, z):
        sub_val = self.step - x
        given_state = self.past_states[:sub_val]
        state_attr = self.get_state_val(given_state, y)

        if y in ['reward_probs', 'alpha']:
            return state_attr
        elif y == 'Q':

            that_action = self.picks[sub_val-1]
            if z == "y":
                q_to_return = state_attr[that_action]
            elif z == "n":
                q_to_return = (state_attr, that_action)

            return q_to_return


    def get_state_val(self, x, y):
        x = tuple(x)
        z = self.all_sequences[x]
        w = getattr(z, y)

        return w

    def logger(self, a_tuple):
        for my_item in a_tuple:
            new_val = eval('self.' + my_item)
            if my_item not in self.log.keys():
                self.log[my_item] = [copy(new_val)]
            else:
                self.log[my_item].append(copy(new_val))


    def create_sequences(self):
        for i in range(self.n_stages + 1):
            choice_combos = list(
                itertools.product(choices, repeat=i))

            if i == 0:
                rwds = [0]
            else:
                rwds = list(stages[i - 1]['reward_probs'].keys())

            n_rwds = len(rwds)
            n_choice_combos = len(choice_combos)
            n_rwd_choice_ratio = n_rwds / self.n_choices

            my_exponent = 0 if i == 0 else 1
            loop_through = (normal_round(n_choice_combos / self.n_choices)) ** my_exponent

            for j in range(loop_through):

                choice_order_shuff = list(range(self.n_choices))
                random.shuffle(choice_order_shuff)

                selected = []
                choice_order_shuff = [0] if i == 0 else choice_order_shuff

                for k, m in enumerate(choice_order_shuff):
                    combo = choice_combos[self.n_choices * j + k]
                    if n_rwd_choice_ratio <= 1:
                        if k % n_rwds == 0:
                            rwds_copy = copy(rwds)
                            np.random.shuffle(rwds_copy)
                        rwd_prob_to_inflate = rwds_copy.pop()
                    else:
                        choose_from = rwds[
                                      normal_round(k * n_rwd_choice_ratio): normal_round((k + 1) * n_rwd_choice_ratio)]
                        rwd_prob_to_inflate = None
                        while rwd_prob_to_inflate not in selected:
                            rwd_prob_to_inflate = np.random.choice(choose_from)
                            selected.append(rwd_prob_to_inflate)

                    if i == 0:
                        added_prob = 0
                    else:
                        added_prob = random.uniform(stages[i - 1]['min_added_r_prob'],
                                                    stages[i - 1]['max_added_r_prob'])

                    choice_combo = Sequence(combo, i,
                                            self.n_choices,
                                            rwd_prob_to_inflate,
                                            added_prob,
                                            n_rwds,
                                            my_exponent)

                    self.all_sequences[combo] = choice_combo

def run_single_softmax_experiment(steps, trials):
    """Run experiment with agent using softmax update rule."""
    print('Running a contextual bandit experiment')

    n_stages = len(stages)
    n_choices = len(choices)

    cols_for_df = ("past_states", "picks", "trial", "step", "choice", "outcome", "outcome_chance", "updated_state", "imm_reward")

    agent = Agent(steps, n_stages, n_choices, cols_for_df)

    for trial in range(trials):
        agent.run(trial)
    df = DataFrame(agent.log, columns=cols_for_df)
    table = pa.Table.from_pandas(df)
    fn = os.path.join('..', '..', 'data', 'softmax_experiment.parquet')
    pq.write_table(table, fn)

    globals().update(locals())  #

    return df


if __name__ == '__main__':
    np.random.seed(42)
    choices = ("left", "right")
    lmbda = .5
    stages = (
        {
            'alpha': .1,
            'beta': .5,
            'likely_trans_prob': .7,
            'reward_probs': {0: 1},
            'min_added_r_prob': 0,
            'max_added_r_prob': 0,
        },
        {
            'alpha': .1,
            'beta': .5,
            'likely_trans_prob': 1.0,
            'reward_probs': {0: .5, 1: .5}, # always arrange these sub-lists in ascending order according its first item
            'min_added_r_prob': .2,
            'max_added_r_prob': .25,
        },
        # {
        #     'alpha': .1,
        #     'beta': .5,
        #     'likely_trans_prob': 1.0,
        #     'reward_probs': {0: .5, 1: .5},
        #     # always arrange these sub-lists in ascending order according its first item
        #     'min_added_r_prob': .2,
        #     'max_added_r_prob': .25,
        # }
    )

    # print('Running experiment with alpha={} and beta={}'.format(alpha, beta))
    trials = 360
    run_single_softmax_experiment(stages, trials)
    # import vis
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # vis.plot_simulation_run()