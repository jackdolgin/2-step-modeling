import numpy as np
import math

def softmax(Q_dict, beta):
    """Compute softmax probabilities for all actions.

    Parameters
    ----------

    Q_dict: dictionary
        Action values for all the actions
    beta: float
        Inverse temperature

    Returns
    -------
    array-like
        Probabilities for each action
    """

    Qs = np.fromiter((i for i in Q_dict.values()), float)
    num = np.exp(Qs * beta)
    return num / num.sum()



def normal_round(n):
    """
    Copied from https://stackoverflow.com/a/41206290/8239878
    """
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)
