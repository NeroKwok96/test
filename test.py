import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.signal import convolve

# C++ translated code
dt = 1 / 25600
acc_df = pd.read_csv('189249000_20231101_035855_acc_time.csv')
vel_df = pd.read_csv('189249000_20231101_035855_vel_time.csv')
acc_arr = acc_df[['vertical', 'horizontal', 'axial']].to_numpy()
vel_arr = vel_df['vertical'].to_numpy()
v2_predicted_arr = np.zeros((25600, 3))
v1 = 0
for i in range(len(acc_arr) - 1):
    a1 = acc_arr[i]
    a2 = acc_arr[i+1] if i < len(acc_arr) - 1 else a1  # Handle the last element case
    delta_acc = (a1 + a2) * 9.81 * 1000
    v2_predicted = v1 + (0.5 * delta_acc * dt)
    v2_predicted_arr[i+1] = v2_predicted  # Start from index 1
    v1 = v2_predicted
# v2_predicted_arr = v2_predicted_arr[:25600]  # Trim the array to desired size

def find_autocorrelation_value(input_data, p):
    coeff = np.zeros(input_data.shape[1])
    # print("coeff: ", coeff)
    for cnt_col in range(input_data.shape[1]):
        # print(cnt_col)
        if p >= 0:
            for n in range(p, input_data.shape[0]):
                coeff[cnt_col] += input_data[n, cnt_col] * input_data[n - p, cnt_col]
        else:
            for n in range(0, input_data.shape[0] + p):
                coeff[cnt_col] += input_data[n, cnt_col] * input_data[n - p, cnt_col]
    # print("coeff: ", coeff)
    return coeff / input_data.shape[0]

# p = 3
# coefficients = find_autocorrelation_value(v2_predicted_arr, 500)
# print("Autocorrelation coefficients:", coefficients)
def find_ar_yule_coefficients(input_data, filter_order):
    yule_matrix_list = []
    yule_vector_list = []
    yule_coeff = np.zeros((filter_order, input_data.shape[1]))

    for _ in range(input_data.shape[1]):
        yule_matrix = np.zeros((filter_order, filter_order))
        yule_vector = np.zeros(filter_order)
        yule_matrix_list.append(yule_matrix)
        yule_vector_list.append(yule_vector)

    for cnt_col in range(filter_order):
        for cnt_row in range(cnt_col, filter_order):
            diff = cnt_row - cnt_col
            coeff = find_autocorrelation_value(input_data, diff)
            for cnt_channel in range(input_data.shape[1]):
                yule_matrix_list[cnt_channel][cnt_row, cnt_col] = coeff[cnt_channel]
                yule_matrix_list[cnt_channel][cnt_col, cnt_row] = coeff[cnt_channel]

    for cnt_row in range(filter_order):
        rhs_coeff = find_autocorrelation_value(input_data, cnt_row + 1)
        for cnt_channel in range(input_data.shape[1]):
            yule_vector_list[cnt_channel][cnt_row] = rhs_coeff[cnt_channel]

    for cnt_channel in range(input_data.shape[1]):
        a = linalg.solve(-yule_matrix_list[cnt_channel], yule_vector_list[cnt_channel])
        for cnt_row in range(a.shape[0]):
            yule_coeff[cnt_row, cnt_channel] = a[cnt_row]

    return yule_coeff
def apply_filter_on_data(input_data, filter_order):
    yule_coeff = find_ar_yule_coefficients(input_data, filter_order)
    filtered_data = np.zeros_like(input_data)
    # print("yule_coeff: ", yule_coeff)
    if input_data.shape[0] > 0 and input_data.shape[1] > 0:
        filtered_data[:, 0] = convolve(input_data[:, 0], yule_coeff[:, 0], mode='same')
        for cnt_col in range(1, input_data.shape[1]):
            filtered_data[:, cnt_col] = convolve(input_data[:, cnt_col], yule_coeff[:, cnt_col], mode='same')
    return filtered_data


filtered_data = apply_filter_on_data(v2_predicted_arr, 10)

from numpy.polynomial import Polynomial


def remove_trend_from_data(input_data, order):
    detrended_data = np.empty_like(input_data)
    number_of_elements = input_data.shape[0]

    if number_of_elements == 0:
        print("The size of input matrix is zero in remove_trend_from_data.")
        detrended_data = np.empty((0, input_data.shape[1]))
    else:
        x_values = np.linspace(0, 1, input_data.shape[0])

        for cnt_col in range(input_data.shape[1]):
            coeff = Polynomial.fit(x_values, input_data[:, cnt_col], order).convert().coef
            poly_y_values = Polynomial(coeff[::-1])(x_values)
            
            detrended_data[:, cnt_col] = input_data[:, cnt_col] - poly_y_values

    return detrended_data
result = remove_trend_from_data(filtered_data, 4)
print(result)
# # Plot the graph of vel_arr, v2_predicted_arr, and filteredData
# time = np.arange(0, len(vel_arr)) * dt  # Create an array of time values
# # plt.plot(time, vel_arr, label='Measured Velocity')
# plt.plot(time, v2_predicted_arr[:, 0], label='Predicted Velocity')  # Index 0 for vertical component
# plt.plot(time, filtered_data[:, 0], label='Filtered Velocity')  # Index 0 for vertical component
# plt.xlabel('Time')
# plt.ylabel('Velocity')
# plt.title('Measured, Predicted, and Filtered Velocity over Time')
# plt.grid(True)
# plt.legend()
# plt.show()