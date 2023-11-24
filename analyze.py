import os
import pandas as pd
import numpy as np
import glob
from dotenv import load_dotenv
import psycopg2
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
xswh_rds_host = os.environ['xswh_RDS_HOST']
xswh_rds_port = os.environ['xswh_RDS_PORT']
xswh_rds_database = os.environ['xswh_RDS_DATABASE']
xswh_rds_user = os.environ['xswh_RDS_USER']
xswh_rds_password = os.environ['xswh_RDS_PASSWORD']

xsdb_rds_host = os.environ['xsdb_RDS_HOST']
xsdb_rds_port = os.environ['xsdb_RDS_PORT']
xsdb_rds_database = os.environ['xsdb_RDS_DATABASE']
xsdb_rds_user = os.environ['xsdb_RDS_USER']
xsdb_rds_password = os.environ['xsdb_RDS_PASSWORD']

# Connect to XS warehouse
conn_to_xswh = psycopg2.connect(
    host=xswh_rds_host,
    port=int(xswh_rds_port),
    database=xswh_rds_database,
    user=xswh_rds_user,
    password=xswh_rds_password
)

conn_to_xsdb = psycopg2.connect(
    host=xsdb_rds_host,
    port=int(xsdb_rds_port),
    database=xsdb_rds_database,
    user=xsdb_rds_user,
    password=xsdb_rds_password
)
# Get All the local dataset
csv_to_process_path = os.path.join('data')
file_path_list = glob.glob(os.path.join(csv_to_process_path, '*', '*', '*', '*', '*'))

fan_csv_path_to_checked = set()
motor_csv_path_to_checked = set()
sensor_data_cache = {}
# Select the latest dataset that used to predict the health score 
# find_latest_prediction_query = """
# WITH cte AS (
#     SELECT health, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp,
#            TO_TIMESTAMP(SPLIT_PART(f.invoked_filename, '_', 2) || SPLIT_PART(f.invoked_filename, '_', 3), 'YYYYMMDDHH24MISS') AS extracted_datetime,
#            ROW_NUMBER() OVER (PARTITION BY machine_name , health_type, sensor_location_name ORDER BY invocation_timestamp DESC) AS rn
#     FROM fact_machine_health f
#     JOIN dim_sensor_info d ON f.sensor_info_id = d.id
#     WHERE sensor_location_name IS NOT NULL AND health_type IS NOT NULL
# )
# SELECT health, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp, extracted_datetime
# FROM cte
# WHERE rn = 1
# ORDER BY machine_name;
# """
cursor_xsdb = conn_to_xsdb.cursor()
sensor_history_query = """
SELECT node_id, location_name
FROM sensor_history sh
JOIN sensor_location sl ON sh.sensor_location_id = sl.id
JOIN sensor s ON s.id = sh.sensor_id
JOIN machine m ON sl.machine_id = m.id
JOIN floorplan f ON f.id = m.floorplan_id
JOIN site ON site.id = f.site_id
JOIN organization o ON o.id = site.organization_id
WHERE site.site_id = 'tswh' and machine_name = '6B ISO Rm 1 Fan no.2'
"""
allowed_fan_sensor_ids = set()
allowed_motor_sensor_ids = set()
# Fetch the sensor data and cache it
sensor_data_cache = {}
# Split sensor ids into fan and motor 
cursor_xsdb.execute(sensor_history_query)
sensor_history_rows = cursor_xsdb.fetchall()
for index in range(len(sensor_history_rows)):
    sensor_id = str(sensor_history_rows[index][0])
    location_name = sensor_history_rows[index][1]
    if location_name == 'Fan-DE':
        allowed_fan_sensor_ids.add(sensor_id)
    if location_name == 'Motor':
        allowed_motor_sensor_ids.add(sensor_id)

# Filter the unwanted dataset path base on the timestamp of the filename in local storage
for file_path in file_path_list:
    file_name = os.path.basename(file_path)
    sensor_id = file_name.split('_')[0]
    formatted_date = file_name.split('_')[1]
    formatted_time = file_name.split('_')[2]
    combined_timestamp = f"{formatted_date}_{formatted_time}"
    if sensor_id in allowed_fan_sensor_ids and (combined_timestamp >= '20231018_000000' and combined_timestamp <= '20231118_000000'):
        fan_csv_path_to_checked.add(file_path)
    elif sensor_id in allowed_motor_sensor_ids and (combined_timestamp >= '20231018_000000' and combined_timestamp <= '20231118_000000'):
        motor_csv_path_to_checked.add(file_path)
fan_csv_paths_sorted = sorted(fan_csv_path_to_checked, key=lambda x: os.path.basename(x).split('_')[1:3])
motor_csv_paths_sorted = sorted(motor_csv_path_to_checked, key=lambda x: os.path.basename(x).split('_')[1:3])

# Debug usage
# for i in fan_csv_paths_sorted:
#     file = os.path.basename(i)
#     print(file)
# for i in motor_csv_paths_sorted:
#     file = os.path.basename(i)
#     print(file)
# print(len(fan_csv_path_to_checked))
# print(len(motor_csv_path_to_checked))
def process_fan_data(csv_path_list):
    for csv in csv_path_list:
        file_name = os.path.basename(csv)
        # print(file_name)
        sensor_id = file_name.split('_')[0]
        data = pd.read_csv(csv)
        data = data.drop(columns=[data.columns[-1]])

        # Fan Balancing Data
        fan_balancing_data = data[(data['frequency (Hz)'] >= 56.3) & (data['frequency (Hz)'] <= 60.3)]

        # Fan Misalignment Data
        fan_misalignment_data_1 = data[(data['frequency (Hz)'] >= 112.6) & (data['frequency (Hz)'] <= 120.6)]
        fan_misalignment_data_2 = data[(data['frequency (Hz)'] >= 168.9) & (data['frequency (Hz)'] <= 180.9)]
        fan_misalignment_data = pd.concat([fan_misalignment_data_1, fan_misalignment_data_2], ignore_index=True)

        # Fan Bearing Data
        fan_bearing_data = data[(data['frequency (Hz)'] >= 140.0) & (data['frequency (Hz)'] <= 1500.0)]
        fan_bearing_squared_data = fan_bearing_data.iloc[:, 1:].apply(lambda x: np.square(x))
        fan_bearing_sum_squared_data = fan_bearing_squared_data.sum(axis=0) / 1.5
        fan_bearing_sqrt_data = np.sqrt(fan_bearing_sum_squared_data)
        # Fan Belt Data
        fan_belt_data_1 = data[(data['frequency (Hz)'] >= 16.0) & (data['frequency (Hz)'] <= 20.0)]
        fan_belt_data_2 = data[(data['frequency (Hz)'] >= 34.0) & (data['frequency (Hz)'] <= 38.0)]
        fan_belt_data_3 = data[(data['frequency (Hz)'] >= 50.0) & (data['frequency (Hz)'] <= 58.0)]
        fan_belt_data_4 = data[(data['frequency (Hz)'] >= 68.0) & (data['frequency (Hz)'] <= 76.0)]
        fan_belt_data_5 = data[(data['frequency (Hz)'] >= 86.0) & (data['frequency (Hz)'] <= 94.0)]
        fan_belt_data = pd.concat([fan_belt_data_1, fan_belt_data_2, fan_belt_data_3, fan_belt_data_4, fan_belt_data_5], ignore_index=True)

        # Fan Flow Data
        fan_flow_data = data[(data['frequency (Hz)'] >= 1394.0) & (data['frequency (Hz)'] <= 1404.0)]
        # print("---------------Fan - Balancing---------------")
        process_other_type_data(fan_balancing_data, sensor_id, 'balancing', file_name)
        # print("---------------Fan - Misalignment---------------")
        process_other_type_data(fan_misalignment_data, sensor_id, 'misalignment', file_name)
        # print("---------------Fan - Bearing---------------")
        process_other_type_data(fan_bearing_sqrt_data, sensor_id, 'bearing', file_name)
        # print("---------------Fan - Belt---------------")
        process_other_type_data(fan_belt_data, sensor_id, 'belt', file_name)
        # print("---------------Fan - Flow---------------")
        process_other_type_data(fan_flow_data, sensor_id, 'flow', file_name)

def process_motor_data(csv_path_list):
    for csv in csv_path_list:
        file_name = os.path.basename(csv)
        # print(file_name)
        sensor_id = file_name.split('_')[0]
        data = pd.read_csv(csv)
        data = data.drop(columns=[data.columns[-1]])

        # Motor Balancing Data
        motor_balancing_data = data[(data['frequency (Hz)'] >= 46) & (data['frequency (Hz)'] <= 50)]

        # Motor Misalignment Data
        motor_misalignment_data_1 = data[(data['frequency (Hz)'] >= 92.0) & (data['frequency (Hz)'] <= 100.0)]
        motor_misalignment_data_2 = data[(data['frequency (Hz)'] >= 142.0) & (data['frequency (Hz)'] <= 150.0)]
        motor_misalignment_data = pd.concat([motor_misalignment_data_1, motor_misalignment_data_2], ignore_index=True)

        # Motor Bearing Data
        motor_bearing_data = data[(data['frequency (Hz)'] >= 140.0) & (data['frequency (Hz)'] <= 1500.0)]
        motor_bearing_squared_data = motor_bearing_data.iloc[:, 1:].apply(lambda x: np.square(x))
        motor_bearing_sum_squared_data = motor_bearing_squared_data.sum(axis=0) / 1.5
        motor_bearing_sqrt_data = np.sqrt(motor_bearing_sum_squared_data)

        # Motor Belt Data
        motor_belt_data_1 = data[(data['frequency (Hz)'] >= 16.0) & (data['frequency (Hz)'] <= 20.0)]
        motor_belt_data_2 = data[(data['frequency (Hz)'] >= 34.0) & (data['frequency (Hz)'] <= 38.0)]
        motor_belt_data_3 = data[(data['frequency (Hz)'] >= 50.0) & (data['frequency (Hz)'] <= 58.0)]
        motor_belt_data_4 = data[(data['frequency (Hz)'] >= 68.0) & (data['frequency (Hz)'] <= 76.0)]
        motor_belt_data_5 = data[(data['frequency (Hz)'] >= 86.0) & (data['frequency (Hz)'] <= 94.0)]
        motor_belt_data = pd.concat([motor_belt_data_1, motor_belt_data_2, motor_belt_data_3, motor_belt_data_4, motor_belt_data_5], ignore_index=True)
        # print("---------------Motor - Balancing---------------")
        process_other_type_data(motor_balancing_data, sensor_id, 'balancing', file_name)
        # print("---------------Motor - Misalignment---------------")
        process_other_type_data(motor_misalignment_data, sensor_id, 'misalignment', file_name)
        # print("---------------Motor - Bearing---------------")
        process_other_type_data(motor_bearing_sqrt_data, sensor_id, 'bearing', file_name)
        # print("---------------Motor - Belt---------------")
        process_other_type_data(motor_belt_data, sensor_id,' belt', file_name)


def process_other_type_data(data, sensor_id, health_type, file_name):
    # Fetch the sensor data from the cache
    if sensor_id in sensor_data_cache:
        selected_row = sensor_data_cache[sensor_id]
        print('hi')
    else:
        select_xsdb_query = """
        SELECT sl.location_name AS sensor_location, m.machine_name, o.subdomain_name AS organization_sym, site.site_id AS site_sym
        FROM sensor s
        JOIN sensor_location sl ON sl.sensor_id = s.id
        JOIN machine m ON sl.machine_id = m.id
        JOIN floorplan f ON f.id = m.floorplan_id
        JOIN site ON site.id = f.site_id
        JOIN organization o ON o.id = site.organization_id
        WHERE s.node_id = %s
        """
        cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))
        selected_row = cursor_xsdb.fetchone()
        sensor_data_cache[sensor_id] = selected_row
    max_vertical = round(data['vertical'].max(), 9)
    max_horizontal = round(data['horizontal'].max(), 9)
    max_axial = round(data['axial'].max(), 9)
    max_velocity = max(max_vertical, max_horizontal, max_axial)
    # print('Health Type:', health_type)
    # print(f"Max Vertical: {max_vertical}")
    # print(f"Max Horizontal: {max_horizontal}")
    # print(f"Max Axial: {max_axial}")
    # print(f"Max Velocity: {max_velocity}")
    print("selected_row: ", sensor_data_cache)
    location_name = selected_row[0] if selected_row else None
    machine_name = selected_row[1] if selected_row else None
    organization_sym = selected_row[2] if selected_row else None
    site_sym = selected_row[3] if selected_row else None
    # print(location_name, machine_name, organization_sym, site_sym)
    cursor_xswh = conn_to_xswh.cursor()
    insert_xswh_query = """
    INSERT INTO staging_machine_health (invoked_filename, max_vertical, max_horizontal, max_axial, max_velocity, computation_type, node_id, sensor_location_name, machine_name, organization_sym, site_sym, health_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    # cursor_xswh.execute(insert_xswh_query, (file_name, max_vertical, max_horizontal, max_axial, max_velocity, 'manual', sensor_id, location_name, machine_name, organization_sym, site_sym, health_type))
    # conn_to_xswh.commit()
    # cursor_xswh.close()

# # Close the cursor after all the function calls
# cursor_xsdb.close()
# Manual computation starts here
process_fan_data(fan_csv_path_to_checked)
process_motor_data(motor_csv_path_to_checked)
print('finished')
print(sensor_data_cache)

# Select the latest results done by manual computation
Latest_records_query = """
WITH cte AS (
     SELECT max_vertical, max_horizontal, max_axial, max_velocity, computation_type, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp,
            TO_TIMESTAMP(SPLIT_PART(f.invoked_filename, '_', 2) || SPLIT_PART(f.invoked_filename, '_', 3), 'YYYYMMDDHH24MISS') AS extracted_datetime,
            ROW_NUMBER() OVER (PARTITION BY machine_name , health_type, sensor_location_name, invoked_filename ORDER BY invocation_timestamp DESC) AS rn
     FROM fact_machine_health f
     JOIN dim_sensor_info d ON f.sensor_info_id = d.id
     WHERE sensor_location_name IS NOT NULL AND health_type IS NOT null and computation_type = 'manual'
 )
 SELECT max_vertical, max_horizontal, max_axial, max_velocity, computation_type, invoked_filename, 
 node_id, sensor_location_name, machine_name, health_type, invocation_timestamp, extracted_datetime, rn
 FROM cte
 WHERE rn = 1
 ORDER BY machine_name;
"""

# # Initialize lists to store the data points
# timestamps = []
# health_scores = []
# health_types = []

# # Iterate over the selected rows
# for row in selected_row:
#     health_score = row[0]
#     node_id = row[2]
#     sensor_location_name = row[3]
#     health_type = row[5]
#     timestamp = row[6]

#     # Add data points to the respective lists
#     timestamps.append(timestamp)
#     health_scores.append(health_score)
#     health_types.append(health_type)

# # Plot the graph
# plt.figure()
# plt.title(sensor_location_name)  # Set the title as the sensor_location_name

# # Define line styles, colors, and markers
# line_styles = ['-', '--', '-.', ':']
# line_colors = ['blue', 'green', 'red', 'purple']
# line_markers = ['o', 's', '^', 'v']

# # Plot each health_type separately with its corresponding data points
# unique_types = set(health_types)
# for i, unique_type in enumerate(unique_types):
#     x = [timestamps[j] for j in range(len(health_types)) if health_types[j] == unique_type]
#     y = [health_scores[j] for j in range(len(health_types)) if health_types[j] == unique_type]
#     plt.plot(x, y, label=unique_type, linestyle=line_styles[i % len(line_styles)], color=line_colors[i % len(line_colors)],
#      )  # Customize line styles, colors, and markers

# plt.xlabel('Timestamp')
# plt.ylabel('Health Score')
# plt.legend()  # Add the legend based on health_type
# plt.tight_layout()  # Improve spacing between subplots
# plt.show()