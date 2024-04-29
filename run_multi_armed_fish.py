import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_armed_fish import MultiArmedFish

def convert_to_dict(results, algo_name):

    results_df = pd.DataFrame({'algorithm' : [algo_name]*len(results),
                                   'catches' : results})
    return results_df

iterations = 500
casts = 300
lures = {'crank bait' : 0.2,
         'spinner' : 0.01,
         'soft plastic' : 0.15,
         'rooster' : 0.10,
         'jerk bait' : 0.05}

fishing_trip = MultiArmedFish(lures, casts)

optimal_results =  fishing_trip.multiple_simulations('optimal', iterations)
optimal_results_df= convert_to_dict(optimal_results, 'Optimal')

eliminate_n_results = fishing_trip.multiple_simulations('eliminate_n', iterations, n = 2, num_tests = 10)
eliminate_n_results_df = convert_to_dict(eliminate_n_results, 'Eliminate N')

ucb_results = fishing_trip.multiple_simulations('ucb', iterations, bound_multiplier = 500)
ucb_results_df = convert_to_dict(ucb_results, 'UCB')

ucb_results = fishing_trip.multiple_simulations('ucb', iterations, bound_multiplier = 2)
ucb_results_10x_df = convert_to_dict(ucb_results, 'UCB 10x')

epsilon_greedy_results = fishing_trip.multiple_simulations('epsilon_greedy', iterations, epsilon = 0.10, schedule = [1, 0.5, 0.1, 0])
epsilon_greedy_results_df = convert_to_dict(epsilon_greedy_results, 'Epsilon Greedy')

one_round_results = fishing_trip.multiple_simulations('one_round_learn', iterations, num_tests = 2)
one_round_results_df = convert_to_dict(one_round_results, 'One Round')



# show results in box plot
final_df = pd.concat([optimal_results_df, 
                      eliminate_n_results_df,
                      ucb_results_df,
                      ucb_results_10x_df,
                      epsilon_greedy_results_df,
                      one_round_results_df])
# make box plot that shows the results
sns.boxplot(x = final_df['catches'],
            y = final_df['algorithm']
            )
# plt.yticks(['', 'Optimal', 'One Round']) 
plt.xlabel('Fish caught')
plt.ylabel('Algorithms')
plt.show()