# Subsampler:
import random
import math

def RobustRasterSampler(dimensions, p):
    '''
        Robustly select which rows/columns are sampled at random with 
        probability p. Ensures every row is scanned at least once 
        before any rows/columns are scanned a second time.
    '''
    n_1, n_2, n_E = dimensions
    mask = np.zeros((n_1, n_2, n_E))
    
    #scan_row = RobustScan(n_E,n2,p)
    scan = np.zeros((n_E, n_2));
    R_C = np.arange(1, n_2, 1)
    q = (p*n_2) % 1;      # Row scan residual.
    numSampleMin = math.floor(p*n_2);   

    '''
    For each energy, only sample (then remove) rows in R_C. Once 
    empty, R_C is refilled to sample rows a second time. If R_C is 
    refilled mid-way through an energy level, R_C is refilled without 
    the rows scanned immediately before. 
    '''
    for i in range(n_E):
        ''' 
        If the number of rows per energy level is not an integer, 
        scan the floor/ceiling at random with prob. q so the expected 
        number of rows is consistent.
        '''
        if random.uniform(0, 1) > q:
            numsample = numSampleMin
        else:
            numsample = numSampleMin+1

        lenR_C = len(R_C)

        if lenR_C <= numsample:
            choice1 = R_C; 
            scan[i,R_C] = 1
            R_C = np.arange(1,n_2, 1)
            R_C = np.delete(R_C, choice)
            choice = np.random.choice(len(R_C),numsample-len(choice1),replace = False)
        else:
            choice = np.random.choice(lenR_C,numsample,replace = False)        
        scan[i,R_C[choice]] = 1
        R_C = np.delete(R_C, choice)

    for i in range(n_1):
        mask[i,:,:] = scan           
    return mask