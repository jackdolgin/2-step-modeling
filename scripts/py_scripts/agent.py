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
from copy import deepcopy

pd.options.display.float_format = '{:.2f}'.format


class Sequence:
    def __init__(self, transitions_to_arrive, stage, n_choices, rwd_prob_to_inflate, added_prob, n_rwds, a_mult, potential_new_states):
        self.stage = stage
        self.transitions_to_arrive = transitions_to_arrive
        self.earlier_states = transitions_to_arrive[:-1]
        self.latest_transition = transitions_to_arrive[-1:]
        self.rand_acts = random.sample(choices, n_choices)
        self.potential_new_states = potential_new_states

        self.rwd_prob_to_inflate = rwd_prob_to_inflate

        self.added_prob = added_prob
        self.n_rwds = n_rwds
        stages_copy = deepcopy(stages)
        for key, val in stages_copy[(self.stage - 1) * a_mult].items(): # so both stage of 0 and of 1 get set to stages[0]
            if key in {'alpha', 'beta', 'w'}:
                if self.stage != len(stages):
                    setattr(self, key, stages_copy[(self.stage) * a_mult][key])
                else:
                    setattr(self, key, stages_copy[(self.stage - 1) * a_mult][key])
            else:
                setattr(self, key, val)

        if hasattr(self, "rwalk_mean"):
            k = 1 / (self.rwalk_max + self.rwalk_min*(self.n_rwds - 1))
            self.rwalk_min *= k # can probably condense this and the next line into one with some kind of for loop
            self.rwalk_max *= k

        self.unlikely_trans_prob = ((1 - self.likely_trans_prob)
                                    / (n_choices - 1))

        for key in self.reward_probs.keys():
            if key == self.rwd_prob_to_inflate:
                self.reward_probs[key] += self.added_prob
            else:
                self.reward_probs[key] -= self.added_prob / (self.n_rwds - 1)

        # based on this setup, the Qtd updates within the agent class, and then the Qmd and Q follow suit within the Sequence class (their values are derivatives of all the Qtd's)
        self.Qtd = {i: np.random.uniform(0, 1e-4) for i in choices} # can think of this `Qtd` more as the Qtd of taking the action from a previous state that attempts to reach this (this object's state + the next state) further-along state
        self.Qmb = {}
        self.Q = {}
        self.q_qmb_update(14)

    # Disappointingly, I couldn't figure out how to account for a setup with rewards at multiple stages :( not sure theoretically how to set it up, cause it would require knowing how much of q at stage n is derived from each of the q's at stage n + 1

    def q_qmb_update(self, x):

        for a_choice in choices:

            if self.stage + 1 >= len(stages):
                mb_and_counting = self.Qtd[a_choice]

            else:
                mb_and_counting = 0

                for a_potential_state in self.potential_new_states[a_choice].values():
                    its_qmbs = a_potential_state[1]
                    biggest_qmb = max(its_qmbs.values())
                    prob_arrival = a_potential_state[0]
                    weighted_value = biggest_qmb * prob_arrival
                    mb_and_counting += weighted_value

            self.Qmb[a_choice] = mb_and_counting

            combined_Q = mb_and_counting * self.w + self.Qtd[a_choice] * (1 - self.w)

            self.Q[a_choice] = combined_Q



    def expected_value(self, x):
        asum = 0
        for some_choice in choices:
            asum += self.potential_new_states[x][some_choice] * self.Qtd[some_choice]
        return asum

    def random_walk(self):
        if hasattr(self, "rwalk_mean"):
            moler = random.gauss(self.rwalk_mean, self.rwalk_sd)

            key_list = list(self.reward_probs.keys())
            akey = random.choice(key_list)

            for key in key_list:
                if key == akey:
                    self.reward_probs[key] += moler
                else:
                    toler = moler / (self.n_rwds - 1) # i put this in the else statement rather than directly below moler because, in the off chance someone only supplies one possible reward and they still ask for a random walk, this else statement will never be reached and this line won't be some number divided by 0
                    self.reward_probs[key] -= toler

                while not self.rwalk_min <= self.reward_probs[key] <= self.rwalk_max:
                    if self.reward_probs[key] > self.rwalk_max:
                        self.reward_probs[key] -= 2 * (self.reward_probs[key] - self.rwalk_max)
                    elif self.reward_probs[key] < self.rwalk_min:
                        self.reward_probs[key] += 2 * (self.rwalk_min - self.reward_probs[key])

class Agent:
    def __init__(self, n_stages, n_choices, cols_for_df):
        self.n_stages = n_stages
        self.n_choices = n_choices
        self.all_sequences = {}
        self.create_sequences()
        self.choice = None

        self.log = {}
        for i in cols_for_df:
            setattr(self, i, None)


    def run(self, trial):
        self.trial = trial
        self.states_so_far = ((),)
        self.picks = []
        self.choose_choice()
        for step in range(self.n_stages):
            self.step = step

            self.transition()

            if self.step < self.n_stages - 1:
                self.choose_choice()
            self.qtd_update()
            self.logger(("imm_reward",), "y")


        [i.random_walk() for i in self.all_sequences.values()]

    def log_rprob_and_qs(self):

        for key, sequence in self.all_sequences.items():
            for i in (("reward_probs", "P"), ("Qtd", "Qtd"), ("Qmb", "Qmb"), ("Q", "Q")):
                for another_key, value in getattr(sequence, i[0]).items():
                    self.logger(((i[1], key, another_key, value),), "n")

    def choose_choice(self):

        self.log_rprob_and_qs()

        self.Qs = self.get_state_val(self.states_so_far, 'Q')
        self.beta = self.get_state_val(self.states_so_far, 'beta')

        p = softmax(self.Qs, self.beta)

        self.choice = np.random.choice(choices, p=p) # to do: make sure this and as many self vars as possible are initialized in the init section

        if len(self.picks) < self.n_stages:
            self.logger(("states_so_far", "picks", "choice"), "y")

        self.picks.append(self.choice)


    def transition(self):

        new_state_likelihoods = self.get_state_val(self.states_so_far, 'potential_new_states')[self.choice] # I think this line will no longer work based on the changes I made in the choose_choice function
        action_probs = {key: val[0] for key, val in new_state_likelihoods.items()}


        self.outcome = np.random.choice(list(action_probs.keys()),
                                        p=list(action_probs.values()))
        self.outcome_freq = action_probs[self.outcome]


        self.states_so_far = self.states_so_far + tuple((self.outcome,))
        self.updated_state = self.states_so_far # this is a little confusing, but I believe the reason we save self.states_so_far as a new variable is because we're saving self.states_so_far twice in each row, once before the transition and once after; if we didn't give it a new name, self.logger would overwwrite the row's first self.states_so_far value with the second rather than saving both

        self.logger(("trial", "step", "outcome", "outcome_freq", "updated_state"), "y")


    def qtd_update(self):

        reward_probs = self.get_state_val(self.states_so_far, 'reward_probs')

        self.imm_reward = np.random.choice(list(reward_probs.keys()),
                                           p=list(reward_probs.values()))

        if self.step < self.n_stages - 1:
            q_1 = self.index_get_state_val(-1, 'Qtd', 'y')
        else:
            q_1 = 0

        q_0 = self.index_get_state_val(0, 'Qtd', 'y')

        self.delta = self.imm_reward + q_1 - q_0

        self.logger(("delta",), "y")

        for i in range(self.step + 1):
            the_Q = self.index_get_state_val(i, 'Qtd', 'n')
            alpha = self.index_get_state_val(i, 'alpha', 'y')
            change = self.delta * alpha * (lmbda ** i)

            the_Q[0][the_Q[1]] += change  # this is where the actual update occurs

        [sequence.q_qmb_update(self.trial) for sequence in self.all_sequences.values()]



    def index_get_state_val(self, x, y, z):
        sub_val = self.step + 1 - x
        given_state = self.states_so_far[:sub_val]             # at any give step, the length of self.states_so_far equals step + 2 (e.g. length of 2 at step 0, at least by the time qtd_update gets called)
        state_attr = self.get_state_val(given_state, y)
        # print("tony")
        if y in ['alpha']:
            return state_attr
        elif y == 'Qtd':
            that_action = self.picks[sub_val - 1]              # likewise, at any given step, the length of self.picks equals step + 2 (e.g. length of 2 at step 0, at least by the time qtd_update gets called); note, however, we use -1 because we're indexing a specific number, not all values prior to an index (i.e. no use of colon in this line's index)
            if z == "y":
                q_to_return = state_attr[that_action]
            elif z == "n":
                q_to_return = (state_attr, that_action)

            return q_to_return


    def get_state_val(self, x, y):
        if len(x) == 1:
            x = tuple(x)
        else:
            x = ((x[0],),) + x[1:]

        z = self.all_sequences[x]
        w = getattr(z, y)

        return w

    def logger(self, a_tuple, x):
        for my_item in a_tuple:
            if x == "y":
                new_val = eval('self.' + my_item)
                jones = my_item
            elif x == "n":
                jones = str(my_item[0]) + '(' + str(my_item[1]) + ", " + str(my_item[2]) + ')'
                new_val = my_item[3]

            if jones not in self.log.keys():
                self.log[jones] = [deepcopy(new_val)]
            else:
                self.log[jones].append(deepcopy(new_val))


    def create_sequences(self):
        big_loop = list(range(self.n_stages + 1))
        big_loop.reverse()
        for i in big_loop:
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
                    if i == 0:
                        combo = tuple([()] + list(combo))
                        veneer = tuple((combo,))  # double check this line, because `combo` might have differences vs. what used to be the input in that function—`self.states_so_far`
                        added_prob = 0
                    else:
                        combo = tuple([((),)] + list(combo))
                        veneer = (combo[0],) + combo[1:]  # ditto the comment four lines above
                        added_prob = random.uniform(stages[i - 1]['min_added_r_prob'],
                                                    stages[i - 1]['max_added_r_prob'])
                    if n_rwd_choice_ratio <= 1:
                        if k % n_rwds == 0:
                            rwds_copy = deepcopy(rwds)
                            np.random.shuffle(rwds_copy)
                        rwd_prob_to_inflate = rwds_copy.pop()
                    else:
                        choose_from = rwds[
                                      normal_round(k * n_rwd_choice_ratio): normal_round((k + 1) * n_rwd_choice_ratio)]
                        rwd_prob_to_inflate = None
                        while rwd_prob_to_inflate not in selected:
                            rwd_prob_to_inflate = np.random.choice(choose_from)
                            selected.append(rwd_prob_to_inflate)


                    potential_new_states = {a_choice: {} for a_choice in choices}

                    if i != self.n_stages:
                        possible_action_pairs = {i: j for i, j in zip(choices,
                                                                      # self.rand_acts)}
                                                                      choices)}

                        for x in choices:

                            likeliest_state = possible_action_pairs[x]

                            for key, sequence in self.all_sequences.items():

                                if sequence.earlier_states == veneer and sequence.stage > 0:

                                    possible_state = sequence.latest_transition[0]

                                    if possible_state == likeliest_state:
                                        prob = sequence.likely_trans_prob
                                    else:
                                        prob = sequence.unlikely_trans_prob
                                    potential_new_states[x][possible_state] = [prob, sequence.Qmb]

                    choice_combo = Sequence(combo, i,
                                            self.n_choices,
                                            rwd_prob_to_inflate,
                                            added_prob,
                                            n_rwds,
                                            my_exponent,
                                            potential_new_states)

                    self.all_sequences[combo] = choice_combo

def run_single_softmax_experiment(trials):
    """Run experiment with agent using softmax update rule."""
    print('Running a contextual bandit experiment')

    n_stages = len(stages)
    n_choices = len(choices)

    cols_for_df = ("states_so_far", "picks", "trial", "step", "choice", "outcome", "outcome_freq", "updated_state", "imm_reward", "delta")

    agent = Agent(n_stages, n_choices, cols_for_df)

    for trial in range(trials):
        agent.run(trial)

    almost_df = {}
    for key, value in agent.log.items():
        if key in ["states_so_far", "updated_state"]:
            alist = []
            for i in value:
                alist.append(str(i))
            almost_df[key] = alist
        else:
            almost_df[key] = value


    df = DataFrame(almost_df, columns=tuple(almost_df.keys()))
    table = pa.Table.from_pandas(df)
    fn = os.path.join('..', '..', 'data', 'softmax_experiment.parquet')
    pq.write_table(table, fn)

    globals().update(locals())  #

    return df


if __name__ == '__main__':
    np.random.seed(41)
    choices = ("left", "right")
    lmbda = 0.57
    stages = (
        {
            'alpha': 0.54,
            'beta': 5.19,
            'w': .39,
            'likely_trans_prob': .7,
            'reward_probs': {0: 1},
            'min_added_r_prob': 0,
            'max_added_r_prob': 0,
        },
        {
            'alpha': 0.42,
            'beta': 3.69,
            'w': 0,
            'likely_trans_prob': 1.0,
            'reward_probs': {0: .5, 1: .5},  # always arrange these sub-lists in ascending order according to its key
            'min_added_r_prob': .2,
            'max_added_r_prob': .25,
            'rwalk_mean': 0,
            'rwalk_sd': .025,
            'rwalk_min': .25,
            'rwalk_max': .75,
        },
    )

    trials = 2000

    run_single_softmax_experiment(trials)
    # import vis
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # vis.plot_simulation_run()
