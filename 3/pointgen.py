from math import sin,cos,log, sqrt
import pandas as pd
import random
import numpy as np

#generates y for x for plotting
def function(*args, gerr=False):
    noise = 0
    if gerr:
        noise = random.gauss(0, 1)
    
    arr = np.array(args)
    return np.sin(arr[:, 0] + arr[:, 1]) + noise

# generates y for x
def function_simple(x, y, gerr=False):
    noise = 0
    if gerr:
        noise = random.gauss(0, 1)
    
    return np.sin(x + y) + noise

#generates our data
def generate_point_set(point_cnt, min, max_x, max_y):
    df = pd.DataFrame(columns=['x', 'y', 'f'])
    x = 0
    y = 0

    for i in range(0, point_cnt):

        x = random.uniform(min, max_x)
        y = random.uniform(min, max_y)

        df.loc[i] = [x, y, function_simple(x, y, False)]
    
    df.to_csv('points.csv', index=False)

# test functions
#generate_point_set()
