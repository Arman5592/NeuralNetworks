from math import sin,cos,log, sqrt
import pandas as pd
import random
import numpy as np

#generates y for x
def function(x, gerr=False):
    noise = 0
    if gerr:
        noise = random.gauss(0, 1)

    value = 2

    return np.piecewise(x, [(x > 1) & (x < 1.5),
                            (x >= 1.5) & (x < 3.5),
                            (x >= 3.5) & (x < 5),
                            (x <= 1) | (x >= 5)], 
                            [lambda x: 6 - 4 * x,
                            lambda x: (x - 1.5)**2,
                            1, 2]) + noise
    

#generates our data
def generate_point_set(point_cnt, min_x, max_x):
    df = pd.DataFrame(columns=['x', 'y'])
    x = 0

    for i in range(0, point_cnt):
        x = random.uniform(min_x, max_x)
        df.loc[i] = [x, function(x, True)]
    
    df.to_csv('points.csv', index=False)

# test functions
#generate_point_set()
