import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import math
import seaborn as sns

from coinflip import *

s = CoinFlipScenario_Betainc(n_pub = 4, n_pri = 4, players = 3)
public_obs = 1
print('# of public flips: %s, # of private flips: %s, # of players: %s' % (s.n_p, s.n_v, s.k))

s.set_public_obs(public_obs)
str_truth = s.truthful_strategy_mode()
print('h_p = %s, truthful strategy is %s\n' % (s.h_p, str_truth))

strat = np.array(str_truth)
plt.figure()
plt.plot(strat, label='truthful')

for k in range(10):
    
    strat_old = strat.copy()
    
    for (signal, report) in enumerate(strat_old):
        # For this signal, find the best report given others playing strat_old
        # Univariate optimization
        # strat[i] = arg max (my_report) s.expected_payoff(private_signal, my_report, strat_old)
        # update strat[i]
        
        def f(my_report):
            return -1 * s.expected_payoff(signal, my_report, strat_old)
        
        result = scipy.optimize.minimize_scalar(f, bounds=(0,1), method='bounded')
        strat[signal] = result.x

    print(strat)
    plt.plot(strat, label=str(k))

plt.legend()            
plt.show()
    
