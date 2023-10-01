from math import sin,cos,log, sqrt
import pandas as pd
import random
import numpy as np

#generates y for x
def function(x, gerr=False):
    if gerr:
        return np.sin(x**2) + random.gauss(0, 1)
    else:
        return np.sin(x**2)

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
