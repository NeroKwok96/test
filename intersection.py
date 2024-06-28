import numpy as np
import pandas as pd

def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    # b1 = A.x - B.x
    b1 = B[0] - A[0]
    c1 = a1*(A[0]) - b1*(A[1])
    
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    # b2 = C.x - D.x
    b2 = D[0] - C[0]
    c2 = a2*(C[0]) - b2*(C[1])
 
    determinant = a1*b2 - a2*b1
 
    if (determinant == 0):
        return (10**9, 10**9)
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = ((a1*c2 - a2*c1)/determinant) * -1
        return (x, y)
 
motor_belt_2x_peak = {'min': 28.4, 'max': 38}
motor_balancing_peak = {'min': 38.4, 'max': 50}
motor_balancing_1x_belt_ratio = 48 / 35.5

df = pd.read_csv('189286845_20240513_144111_vel_freq.csv')
df_in_range_motor_belt_2x_peak = df[(df['frequency (Hz)'] >= motor_belt_2x_peak['min']) & (df['frequency (Hz)'] <= motor_belt_2x_peak['max'])][['frequency (Hz)', 'vertical']].nlargest(1, 'vertical')
df_in_range_motor_balancing_peak = df[(df['frequency (Hz)'] >= motor_balancing_peak['min']) & (df['frequency (Hz)'] <= motor_balancing_peak['max'])][['frequency (Hz)', 'vertical']].nlargest(1, 'vertical')
print(df_in_range_motor_belt_2x_peak['frequency (Hz)'].values[0])
print(df_in_range_motor_balancing_peak['frequency (Hz)'].values[0])
dataset_peak_ratio = df_in_range_motor_balancing_peak['frequency (Hz)'].values[0] / df_in_range_motor_belt_2x_peak['frequency (Hz)'].values[0]

if (dataset_peak_ratio < motor_balancing_1x_belt_ratio + 0.05
    or dataset_peak_ratio > motor_balancing_1x_belt_ratio - 0.05):
    print('Peak Shift')
    
    df_in_range_motor_balancing = df[(df['frequency (Hz)'] >= motor_balancing_peak['min']) & (df['frequency (Hz)'] <= motor_balancing_peak['max'])][['frequency (Hz)', 'vertical']].nlargest(5, 'vertical')
    min_diff = 100
    motor_balancing_index = 0

    for i in df_in_range_motor_balancing['frequency (Hz)'].index:
        diff = (df.iloc[i]['frequency (Hz)'] / df_in_range_motor_belt_2x_peak['frequency (Hz)'].values[0]) - motor_balancing_1x_belt_ratio
        if diff < min_diff and diff > 0:
            min_diff = diff
            motor_balancing_index = i
        
if df.iloc[motor_balancing_index - 1]['vertical'] > df.iloc[motor_balancing_index + 1]['vertical']:
    print('method 1')
    A = (df.iloc[motor_balancing_index - 2]['frequency (Hz)'], df.iloc[motor_balancing_index - 2]['vertical'])
    B = (df.iloc[motor_balancing_index - 1]['frequency (Hz)'], df.iloc[motor_balancing_index - 1]['vertical'])
    C = (df.iloc[motor_balancing_index]['frequency (Hz)'], df.iloc[motor_balancing_index]['vertical'])
    D = (df.iloc[motor_balancing_index + 1]['frequency (Hz)'], df.iloc[motor_balancing_index + 1]['vertical'])
else:
    print('method 2')
    A = (df.iloc[motor_balancing_index - 1]['frequency (Hz)'], df.iloc[motor_balancing_index - 1]['vertical'])
    B = (df.iloc[motor_balancing_index]['frequency (Hz)'], df.iloc[motor_balancing_index]['vertical'])
    C = (df.iloc[motor_balancing_index + 1]['frequency (Hz)'], df.iloc[motor_balancing_index + 1]['vertical'])
    D = (df.iloc[motor_balancing_index + 2]['frequency (Hz)'], df.iloc[motor_balancing_index + 2]['vertical'])
    
# # Driver code
# A = (37.5, 0.126495)
# B = (39.0625, 2.36113)
# C = (40.625, 3.990084)
# D = (42.1875, 1.64468)
 
intersection = lineLineIntersection(A, B, C, D)
print(intersection)