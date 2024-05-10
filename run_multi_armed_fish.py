import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_armed_fish import MultiArmedFish

# set up attributes for MultiArmedFish instance
iterations = 500
casts = 300
lures = {'crank bait' : 0.2,
         'spinner' : 0.01,
         'soft plastic' : 0.15,
         'rooster' : 0.10,
         'jerk bait' : 0.05}

fishing_trip = MultiArmedFish(lures, casts)

# create all the algorithms to run
algorithm_runs = [{'name' : 'Optimal',
                   'strategy' : 'optimal',
                   'iterations' : iterations},
                   {'name' : 'Random',
                   'strategy' : 'random',
                   'iterations' : iterations},
                   {'name' : 'One -Round',
                    'strategy' : 'one_round_learn',
                   'num_tests' : 4,
                   'iterations' : iterations
                   },
                {'name' : 'Epsilon - 0.10, 35',
                   'strategy' : 'epsilon_greedy',
                   'epsilon' : 0.10,
                   'iterations' : iterations,
                   'rand_start_casts' : 35
                   },            
                  {'name' : 'Iter. Elimination',
                   'strategy' : 'eliminate_n',
                   'n' : 2,
                   'num_tests' : 3,
                   'iterations' : iterations
                   }, 
                    {'name' : 'UCB',
                   'strategy' : 'ucb',
                   'bound_multiplier' : 0.1,
                   'iterations' : iterations
                   }                                                                                                                                                                                                                            
                   ]

def convert_to_dict(results, algo_name):

    '''Convert simulation output to dataframe

       inputs:
            results (list)  : list of bools representing
                              if a fish was caught on a cast
            algo_name (str) : name of algorithm that will be in
                              dataframe and later used in visualization
        
       outputs:
            results_df (dataframe) : df with two columns, one with
                                     algorithm name, another with 
                                     boolean of success for each cast
    '''

    results_df = pd.DataFrame({'algorithm' : [algo_name]*len(results),
                                   'catches' : results})
    return results_df


def compare_algorithms(trip_instance, algorithm_runs):

    '''
        Runs multiple algorithms based on user inputs
        and make a boxplot to compare the runs.

        inputs:
            trip_instance (MultiArmedFish) : instance of MultiArmedFish.
            algorithm_runs (list)          : list of dictionaries - each 
                                             element/dictionary has inputs
                                             for multiple_simulations method
    '''

    solution_df_list = []

    for run_args in algorithm_runs:

        temp_name = run_args.pop('name')

        temp_result = trip_instance.multiple_simulations(**run_args)
        temp_result_df = convert_to_dict(temp_result, temp_name)
        solution_df_list.append(temp_result_df)

    final_df = pd.concat(solution_df_list)

    # make box plot that shows the results
    sns.boxplot(x = final_df['catches'],
                y = final_df['algorithm']
                )
    # plt.yticks(['', 'Optimal', 'One Round']) 
    plt.xlabel('Fish caught', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Algorithms', fontsize=14)
    plt.yticks(fontsize=12)
    plt.show()    

    return

# compare algorithms
compare_algorithms(fishing_trip, algorithm_runs)