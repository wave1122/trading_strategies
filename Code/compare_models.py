# ======================================= Compare the best of several models with a benchmark model by bootstrap ================================================= #
#============================================================================================================================================== #
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List
import os
import time

rng = np.random.RandomState(seed = 1234)

# Generate M sets of random observation indices used for Politis & Romano's (1994) stationary bootstrap
def gen_indices_SB(  N: int, # the bootstrap sample size
                                    M: int = 499, # the number of sets of random observation indices
                                    block: int = 100, # the mean block length
                                ):
    """ Generate a M by N array of random observation indices.
    """
    assert(N >= 2*block), "the sample size must be twice bigger than the mean block length!"
    
    q = 1./block
    indices_mat = [[] for _ in range(M)]
    for i in np.arange(M):
        theta0 = rng.randint(0, N) # generate the first index of a set of random observation indices
        indices_mat[i].append(theta0)
        for j in np.arange(1, N):
            U = rng.uniform(0., 1.)
            if U < q:
                theta1 = rng.randint(0, N)
            else:
                theta1 = theta0 + 1
                if theta1 > N - 1:
                    theta1 = 0
            indices_mat[i].append(theta1)
            theta0 = theta1 # reset 'theta0'
    return np.array(indices_mat)

def do_reality_check_SB(  perf_A: np.ndarray, # a N by K array of N values on each of K performance measures
                                            perf_B: np.ndarray, # a N by 1 array of  N values on a benchmark performance measure
                                            M: int = 499, # the number of sets of random observation indices
                                            block: int = 100, # the mean block length
                                        ) -> list:
    """ Calculate the Reality Check (RC) p-values.
    INPUT
        perf_A: a N by K array of N values on each of K performance measures
        perf_B: a N by 1 array of  N values on a benchmark performance measure
        M: the number of sets of random observation indices
        block: the mean block length
    OUTPUT
        a list of RC p-values
    """
    N = perf_A.shape[0]
    K = perf_A.shape[1]
    diff = perf_A - perf_B
    
    # start with a model
    ## compute the sample statistic
    V0 = np.sqrt(N) * np.mean(diff[:, 0])
        
    ## compute the bootstrap statistic for the first model added
    indices_mat = gen_indices_SB(N, M, block)
    # print('indices_mat = ', indices_mat)
    V0_bs, p_vals = [], []
    for i in np.arange(M):
        V_i = np.sqrt(N) * (np.mean(diff[indices_mat[i,:], 0]) - np.mean(diff[:, 0]))
        V0_bs.append(V_i)
    V0_bs = np.array(V0_bs)
    p_vals.append( len(V0_bs[V0_bs > V0]) / M )
    
    # sequentially add a model
    for j in np.arange(1, K):
        ## compute the sample statistic
        V1 = max(V0, np.sqrt(N) * np.mean(diff[:, j]))
        
        ## compute the bootstrap statistic
        V1_bs = []
        for i in np.arange(M):
            V_i = max( V0_bs[i], np.sqrt(N)*(np.mean(diff[indices_mat[i,:], j]) - np.mean(diff[:, j])) )
            V1_bs.append(V_i)
        V1_bs = np.array(V1_bs)
        p_vals.append( len(V1_bs[V1_bs > V1]) / M )
        
        V0 = V1 # reset 'V0'
        V0_bs = V1_bs # reset 'V0_bs'
        
    return p_vals

# if __name__ == "__main__":
#     startTime = time.time()
    
#     # indices_mat = gen_indices_SB( N = 20, # the bootstrap sample size
#     #                                                     M= 4, # the number of sets of random observation indices
#     #                                                     block = 5, # the mean block length
#     #                                                 )
#     # print(indices_mat)
#     # print(indices_mat.shape)
#     # print(indices_mat[0, :])
    
#     N = 20
#     K = 5
#     M = 4
#     block = 5
#     perf_A = rng.normal(0., 1., size = (N, K))
#     perf_B = rng.normal(0., 1., size = (N, 1))
#     p_values = do_reality_check_SB(  perf_A, # a N by K array of N values on each of K performance measures
#                                             perf_B, # a N by 1 array of  N values on a benchmark performance measure
#                                             M = M, # the number of sets of random observation indices
#                                             block = block, # the mean block length
#                                         )
#     print(p_values)
    
#     print( 'The script took {} second !'.format(time.time() - startTime) )
                
                
            
    
    
