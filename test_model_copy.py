import os
import tensorflow as tf
import pandas as pd
import numpy as np
from models.low_vel_gan import LowVelGAN
import glob
from dotenv import load_dotenv
import psycopg2
from datetime import datetime

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

# Model Path
fan_balancing_model_path = "out/models/balancing_fan_model_20231019A"
fan_misalignment_model_path = "out/models/misalignment_fan_model_20231019A"
fan_belt_model_path = "out/models/belt_fan_model_20231019A"
fan_bearing_model_path = "out/models/bearing_fan_model_20231031A"
fan_flow_model_path = "out/models/flow_fan_model_20231102B"
motor_balancing_model_path = "out/models/balancing_motor_model_20231019A"
motor_misalignment_model_path = "out/models/misalignment_motor_model_20231019A"
motor_belt_model_path = "out/models/belt_motor_model_20231019A"
motor_bearing_model_path = "out/models/bearing_motor_model_20231106A"

# Fan threshold
fan_balancing_threshold = [0.11895789301461002, 0.261937856117473, 0.4525778069212904]
fan_misalignment_threshold = [0.02076109964874931, 0.04311671235004436, 0.0729241959517711]
fan_belt_threshold = [0.11747596569508362, 0.22611403071194214, 0.3709647840677535]
fan_bearing_threshold = [6.090199449930493, 11.083541999094127, 17.74133206464564]
fan_flow_threshold = [0.012982298065879062, 0.026109734784352082, 0.043612983742316105]

# Motor threshold
motor_balancing_threshold = [0.3501516760306645, 0.7249693120315679, 1.2247261600327726]
motor_misalignment_threshold = [0.13641731226195508, 0.2982093459232462, 0.5139320574716343]
motor_belt_threshold = [0.33105875518823297, 0.6851169317847347, 1.1571945005800701]
motor_bearing_threshold = [5.9299586040158445, 12.78388486885658, 21.92245322197756]

def calculate_machine_health(mae, threshold_good, threshold_usable, threshold_unsatisfactory):
    if mae >= 0 and mae < threshold_good:
        machine_health = 1
    elif mae > threshold_good and mae < threshold_usable:
        machine_health = 2
    elif mae > threshold_usable and mae < threshold_unsatisfactory:
        machine_health = 3
    else:
        machine_health = 4
    return machine_health

# Connect to XS warehouse
conn_to_xswh = psycopg2.connect(
    host=xswh_rds_host,
    port=int(xswh_rds_port),
    database=xswh_rds_database,
    user=xswh_rds_user,
    password=xswh_rds_password
)
# Connect to XS database
conn_to_xsdb = psycopg2.connect(
    host=xsdb_rds_host,
    port=int(xsdb_rds_port),
    database=xsdb_rds_database,
    user=xsdb_rds_user,
    password=xsdb_rds_password
)

cursor_xswh = conn_to_xswh.cursor()
cursor_xsdb = conn_to_xsdb.cursor()

# Get all files
csv_to_process_path = os.path.join('data')
file_path_list = glob.glob(os.path.join(csv_to_process_path, '*', '*', '*', '*', '*'))
fan_csv_path_to_checked = set()
motor_csv_path_to_checked = set()

# Limited date range:
customized_period_from = '20231018_000000'
customized_period_to = '20231118_000000'

# sensor_history_query = """
# SELECT node_id, location_name
# FROM sensor_history sh
# JOIN sensor_location sl ON sh.sensor_location_id = sl.id
# JOIN sensor s ON s.id = sh.sensor_id
# JOIN machine m ON sl.machine_id = m.id
# JOIN floorplan f ON f.id = m.floorplan_id
# JOIN site ON site.id = f.site_id
# JOIN organization o ON o.id = site.organization_id
# WHERE site.site_id = 'tswh' and machine_name = '6B ISO Rm 1 Fan no.2'
# """
sensor_history_query = """
SELECT node_id, location_name, m.machine_name, o.subdomain_name, site.site_id, period_from, period_to
FROM sensor_history sh
JOIN sensor_location sl ON sh.sensor_location_id = sl.id
JOIN sensor s ON s.id = sh.sensor_id
JOIN machine m ON sl.machine_id = m.id
JOIN floorplan f ON f.id = m.floorplan_id
JOIN site ON site.id = f.site_id
JOIN organization o ON o.id = site.organization_id
where site.site_id = 'tswh' and machine_name in ('6B ISO Rm 1 Fan no.1', '6B ISO Rm 1 Fan no.2', '6B ISO Rm 2 Fan no.1', '6B ISO Rm 2 Fan no.2')
"""
allowed_fan_sensor_ids = set()
allowed_motor_sensor_ids = set()
invoked_filename_set = set()

# Fetch the sensor data and cache it
sensor_data_cache = {}
sensor_data_cache_list = []

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
    sensor_data_cache_list.append(sensor_history_rows[index])
# Filter the unwanted dataset path base on the timestamp of the filename in local storage
for file_path in file_path_list:
    file_name = os.path.basename(file_path)
    sensor_id_from_file = file_name.split('_')[0]
    formatted_date = file_name.split('_')[1]
    formatted_time = file_name.split('_')[2]
    combined_timestamp = f"{formatted_date}_{formatted_time}"
    for sensor_data in sensor_data_cache_list:
        sensor_id_from_history = str(sensor_data[0])
        period_from = sensor_data[5].strftime("%Y%m%d_%H%M%S")
        period_to = sensor_data[6]
        if period_to == None: period_to = datetime.now().strftime("%Y%m%d_%H%M%S")
        else: period_to = sensor_data[6].strftime("%Y%m%d_%H%M%S")
        if sensor_id_from_file in allowed_fan_sensor_ids and (sensor_id_from_file == sensor_id_from_history) and (combined_timestamp >= customized_period_from and combined_timestamp >= period_from) and (combined_timestamp <= period_to and combined_timestamp <= customized_period_to):
            fan_csv_path_to_checked.add(file_path)
        elif sensor_id_from_file in allowed_motor_sensor_ids and (sensor_id_from_file == sensor_id_from_history) and (combined_timestamp >= customized_period_from and combined_timestamp >= period_from) and (combined_timestamp <= period_to and combined_timestamp <= customized_period_to):
            motor_csv_path_to_checked.add(file_path)

# Sort the csv by date in ascending order
fan_csv_paths_sorted = sorted(fan_csv_path_to_checked, key=lambda x: os.path.basename(x).split('_')[1:3])
motor_csv_paths_sorted = sorted(motor_csv_path_to_checked, key=lambda x: os.path.basename(x).split('_')[1:3])
fan_dataset = set()
motor_dataset = set()
# Debug usage
fan_sensor_id_validation = set()
motor_sensor_id_validation = set()
for i in fan_csv_paths_sorted:
    file = os.path.basename(i)
    sensor_id_from_file = file.split('_')[0]
    fan_sensor_id_validation.add(sensor_id_from_file)
    fan_dataset.add(file)
for i in motor_csv_paths_sorted:
    file = os.path.basename(i)
    sensor_id_from_file = file.split('_')[0]
    motor_sensor_id_validation.add(sensor_id_from_file)
    motor_dataset.add(file)
print("Number of dataset of fan: ", len(fan_dataset))
print("Number of dataset of motor: ", len(motor_dataset))

def preprocessing(csv_path_list):
    # Skip the predicted file_name:
    temp_query = """
    WITH cte AS (
         SELECT mae, health, computation_type, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp,
                TO_TIMESTAMP(SPLIT_PART(f.invoked_filename, '_', 2) || SPLIT_PART(f.invoked_filename, '_', 3), 'YYYYMMDDHH24MISS') AS extracted_datetime,
                ROW_NUMBER() OVER (PARTITION BY machine_name , health_type, sensor_location_name, invoked_filename ORDER BY invocation_timestamp DESC) AS rn
         FROM fact_machine_health f
         JOIN dim_sensor_info d ON f.sensor_info_id = d.id
         WHERE sensor_location_name IS NOT NULL AND health_type IS NOT NULL and machine_name in ('6B ISO Rm 1 Fan no.1', '6B ISO Rm 1 Fan no.2', '6B ISO Rm 2 Fan no.1', '6B ISO Rm 2 Fan no.2')
     )
     SELECT mae, health, computation_type, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp, extracted_datetime
     FROM cte
     WHERE rn = 1 and computation_type = 'model' and extracted_datetime >= '2023-10-18' and extracted_datetime <= '2023-11-18'
     ORDER BY extracted_datetime ASC;
    """
    # cursor_xswh.execute(temp_query)
    # temp_query_result = cursor_xswh.fetchall()

    # Start the data processing
    for csv in csv_path_list:
        file_name = os.path.basename(csv)
        print(f"Processing: {file_name}")
        data = pd.read_csv(csv)
        data = data.drop(columns=[data.columns[-1]])
        sensor_id = file_name.split('_')[0]
        # Fan balancing
        balancing_df = data.to_numpy()[:128, 1:3]
        balancing_squared_values = np.square(balancing_df)
        balancing_input_data = np.concatenate([balancing_df, balancing_squared_values], axis=-1)
        balancing_actual_data = np.reshape(balancing_input_data, (1,128,4,1))
        # Fan misalignment
        misalignment_df = data.to_numpy()[100:324, 3:]
        misalignment_squared_values = np.square(misalignment_df)
        misalignment_input_data = np.concatenate([misalignment_df, misalignment_squared_values], axis=-1)
        misalignment_actual_data = np.reshape(misalignment_input_data, (1,224,2,1))
        # Fan belt
        belt_df = data.to_numpy()[:128, 1:]
        belt_squared_values = np.square(belt_df[:, 0:2])
        belt_vertical_variance = np.abs(belt_df[:, 0:1] - np.max(belt_df[:, 0:1]))
        belt_horizontal_variance = np.abs(belt_df[:, 1:2] - np.max(belt_df[:, 1:2]))
        belt_axial_variance = np.abs(belt_df[:, 2:] - np.max(belt_df[:, 2:]))
        belt_input_data = np.concatenate([belt_df, belt_squared_values, belt_vertical_variance, belt_horizontal_variance, belt_axial_variance], axis=-1)
        belt_actual_data = np.reshape(belt_input_data, (1,128,8,1))
        # Fan flow
        flow_df_1 = data.to_numpy()[885:885 + 42, 1:3]
        flow_df_2 = data.to_numpy()[1781:1781 + 43, 1:3]
        flow_df_3 = data.to_numpy()[2677:2677 + 43, 1:3]
        flow_input_data = np.concatenate([flow_df_1, flow_df_2, flow_df_3], axis=0)
        flow_actual_data = np.reshape(flow_input_data, (1,128,2,1))
        # Fan bearing
        bearing_df = data.to_numpy()[:2048, 1:]
        bearing_squared_values = np.square(bearing_df[:, 0:2])
        bearing_vertical_variance = np.abs(bearing_df[:, 0:1] - np.sqrt(np.sum(np.square(bearing_df[:, 0:1])) / 1.5))
        bearing_horizontal_variance = np.abs(bearing_df[:, 1:2] - np.sqrt(np.sum(np.square(bearing_df[:, 1:2])) / 1.5))
        bearing_axial_variance = np.abs(bearing_df[:, 2:] - np.sqrt(np.sum(np.square(bearing_df[:, 2:])) / 1.5))
        bearing_input_data = np.concatenate([bearing_df, bearing_squared_values, bearing_vertical_variance, bearing_horizontal_variance, bearing_axial_variance], axis=-1)
        bearing_actual_data = np.reshape(bearing_input_data, (1,2048,8,1))

        fan_record_count = motor_record_count = 0
        check_existed_query = """SELECT COUNT(DISTINCT health_type) FROM staging_machine_health WHERE invoked_filename = %s AND health_type IN ('balancing', 'misalignment', 'belt', 'bearing', 'flow') AND sensor_location_name = 'Fan-DE' AND computation_type = 'model'"""
        cursor_xswh.execute(check_existed_query, (file_name,))
        fan_record_count = cursor_xswh.fetchone()

        check_existed_query = """SELECT COUNT(DISTINCT health_type) FROM staging_machine_health WHERE invoked_filename = %s AND health_type IN ('balancing', 'misalignment', 'belt', 'bearing') AND sensor_location_name = 'Motor' AND computation_type = 'model'"""
        cursor_xswh.execute(check_existed_query, (file_name,))
        motor_record_count = cursor_xswh.fetchone()

        computation_type = None
        if sensor_id in fan_sensor_id_validation:
            # for result in temp_query_result:
            if (computation_type != 'model' or computation_type == None) and fan_record_count[0] != 5:
                fan_dataset_prediction(balancing_actual_data, misalignment_actual_data, belt_actual_data, flow_actual_data, bearing_actual_data, file_name, sensor_id)
        elif sensor_id in motor_sensor_id_validation:
            # for result in temp_query_result:
            if (computation_type != 'model' or computation_type == None) and motor_record_count[0] != 4:
                motor_dataset_prediction(balancing_actual_data, misalignment_actual_data, belt_actual_data, bearing_actual_data, file_name, sensor_id)
        print(f"{file_name} already predicted, skipping")


def fan_dataset_prediction(balancing_actual_data, misalignment_actual_data, belt_actual_data, flow_actual_data, bearing_actual_data, file_name, sensor_id):
    # Load Fan models
    fan_balancing_model = LowVelGAN(input_shape=balancing_actual_data.shape)
    fan_balancing_model.load_keras_models(save_dir=fan_balancing_model_path)
    fan_balancing_prediction_result = fan_balancing_model.generator.predict(balancing_actual_data)[1]
    fan_misalignment_model = LowVelGAN(input_shape=misalignment_actual_data.shape)
    fan_misalignment_model.load_keras_models(save_dir=fan_misalignment_model_path)
    fan_misalignment_prediction_result = fan_misalignment_model.generator.predict(misalignment_actual_data)[1]
    fan_belt_model = LowVelGAN(input_shape=belt_actual_data.shape)
    fan_belt_model.load_keras_models(save_dir=fan_belt_model_path)
    fan_belt_prediction_result = fan_belt_model.generator.predict(belt_actual_data)[1]
    fan_flow_model = LowVelGAN(input_shape=flow_actual_data.shape)
    fan_flow_model.load_keras_models(save_dir=fan_flow_model_path)
    fan_flow_prediction_result = fan_flow_model.generator.predict(flow_actual_data)[1]
    fan_bearing_model = LowVelGAN(input_shape=bearing_actual_data.shape)
    fan_bearing_model.load_keras_models(save_dir=fan_bearing_model_path)
    fan_bearing_prediction_result = fan_bearing_model.generator.predict(bearing_actual_data)[1]

    # Fan mae
    fan_balancing_mae = round(np.mean(np.absolute(np.subtract(balancing_actual_data, fan_balancing_prediction_result))), 10)
    fan_misalignment_mae = round(np.mean(np.absolute(np.subtract(misalignment_actual_data, fan_misalignment_prediction_result))), 10)
    fan_belt_mae = round(np.mean(np.absolute(np.subtract(belt_actual_data, fan_belt_prediction_result))), 10)
    fan_flow_mae = round(np.mean(np.absolute(np.subtract(flow_actual_data, fan_flow_prediction_result))), 10)
    fan_bearing_mae = round(np.mean(np.absolute(np.subtract(bearing_actual_data, fan_bearing_prediction_result))), 10)
    fan_balancing_machine_health = calculate_machine_health(fan_balancing_mae, fan_balancing_threshold[0], fan_balancing_threshold[1], fan_balancing_threshold[2])
    fan_misalignment_machine_health = calculate_machine_health(fan_misalignment_mae, fan_misalignment_threshold[0], fan_misalignment_threshold[1], fan_misalignment_threshold[2])
    fan_belt_machine_health = calculate_machine_health(fan_belt_mae, fan_belt_threshold[0], fan_belt_threshold[1], fan_belt_threshold[2])
    fan_flow_machine_health = calculate_machine_health(fan_flow_mae, fan_flow_threshold[0], fan_flow_threshold[1], fan_flow_threshold[2])
    fan_bearing_machine_health = calculate_machine_health(fan_bearing_mae, fan_bearing_threshold[0], fan_bearing_threshold[1], fan_bearing_threshold[2])
    print('fan_balancing_mae: ', fan_balancing_mae)
    print('fan_balancing_machine_health: ', fan_balancing_machine_health)
    print('fan_misalignment_mae: ', fan_misalignment_mae)
    print('fan_misalignment_machine_health: ', fan_misalignment_machine_health)
    print('fan_belt_mae: ', fan_belt_mae)
    print('fan_belt_machine_health: ', fan_belt_machine_health)
    print('fan_flow_mae: ', fan_flow_mae)
    print('fan_flow_machine_health: ', fan_flow_machine_health)
    print('fan_bearing_mae: ', fan_bearing_mae)
    print('fan_bearing_machine_health: ', fan_bearing_machine_health)
    
    for sensor_data in sensor_data_cache_list:
        sensor_id_from_history = str(sensor_data[0])
        period_from = sensor_data[5].strftime("%Y%m%d_%H%M%S")
        period_to = sensor_data[6]
        if period_to == None: period_to = datetime.now().strftime("%Y%m%d_%H%M%S")
        else: period_to = sensor_data[6].strftime("%Y%m%d_%H%M%S")
        if sensor_id_from_history == sensor_id and customized_period_from >= period_from and customized_period_from <= period_to:
            print(sensor_data)
            sensor_location = sensor_data[1]
            machine_name = sensor_data[2]
            organization_sym = sensor_data[3]
            site_sym = sensor_data[4]

            # Insert the predicted results into warehouse
            cursor_xswh = conn_to_xswh.cursor()
            insert_xswh_query = """
            INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_balancing_mae, fan_balancing_machine_health, organization_sym, site_sym, file_name, 'balancing', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_misalignment_mae, fan_misalignment_machine_health, organization_sym, site_sym, file_name, 'misalignment', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_belt_mae, fan_belt_machine_health, organization_sym, site_sym, file_name, 'belt', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_flow_mae, fan_flow_machine_health, organization_sym, site_sym, file_name, 'flow', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_bearing_mae, fan_bearing_machine_health, organization_sym, site_sym, file_name, 'bearing', 'model'))
            conn_to_xswh.commit()

def motor_dataset_prediction(balancing_actual_data, misalignment_actual_data, belt_actual_data, bearing_actual_data, file_name, sensor_id):
    # Load motor models
    motor_balancing_model = LowVelGAN(input_shape=balancing_actual_data.shape)
    motor_balancing_model.load_keras_models(save_dir=motor_balancing_model_path)
    motor_balancing_prediction_result = motor_balancing_model.generator.predict(balancing_actual_data)[1]
    motor_misalignment_model = LowVelGAN(input_shape=misalignment_actual_data.shape)
    motor_misalignment_model.load_keras_models(save_dir=motor_misalignment_model_path)
    motor_misalignment_prediction_result = motor_misalignment_model.generator.predict(misalignment_actual_data)[1]
    motor_belt_model = LowVelGAN(input_shape=belt_actual_data.shape)
    motor_belt_model.load_keras_models(save_dir=motor_belt_model_path)
    motor_belt_prediction_result = motor_belt_model.generator.predict(belt_actual_data)[1]
    motor_bearing_model = LowVelGAN(input_shape=bearing_actual_data.shape)
    motor_bearing_model.load_keras_models(save_dir=motor_bearing_model_path)
    motor_bearing_prediction_result = motor_bearing_model.generator.predict(bearing_actual_data)[1]

    # Motor mae
    motor_balancing_mae = np.mean(np.absolute(np.subtract(balancing_actual_data, motor_balancing_prediction_result)))
    motor_misalignment_mae = np.mean(np.absolute(np.subtract(misalignment_actual_data, motor_misalignment_prediction_result)))
    motor_belt_mae = np.mean(np.absolute(np.subtract(belt_actual_data, motor_belt_prediction_result)))
    motor_bearing_mae = np.mean(np.absolute(np.subtract(bearing_actual_data, motor_bearing_prediction_result)))
    motor_balancing_machine_health = calculate_machine_health(motor_balancing_mae, motor_balancing_threshold[0], motor_balancing_threshold[1], motor_balancing_threshold[2])
    motor_misalignment_machine_health = calculate_machine_health(motor_misalignment_mae, motor_misalignment_threshold[0], motor_misalignment_threshold[1], motor_misalignment_threshold[2])
    motor_belt_machine_health = calculate_machine_health(motor_belt_mae, motor_belt_threshold[0], motor_belt_threshold[1], motor_belt_threshold[2])
    motor_bearing_machine_health = calculate_machine_health(motor_bearing_mae, motor_bearing_threshold[0], motor_bearing_threshold[1], motor_bearing_threshold[2])
    print('motor_balancing_mae: ', motor_balancing_mae)
    print('motor_balancing_machine_health: ', motor_balancing_machine_health)
    print('motor_misalignment_mae: ', motor_misalignment_mae)
    print('motor_misalignment_machine_health: ', motor_misalignment_machine_health)
    print('motor_belt_mae: ', motor_belt_mae)
    print('motor_belt_machine_health: ', motor_belt_machine_health)
    print('motor_bearing_mae: ', motor_bearing_mae)
    print('motor_bearing_machine_health: ', motor_bearing_machine_health)

    for sensor_data in sensor_data_cache_list:
        sensor_id_from_history = str(sensor_data[0])
        period_from = sensor_data[5].strftime("%Y%m%d_%H%M%S")
        period_to = sensor_data[6]
        if period_to == None: period_to = datetime.now().strftime("%Y%m%d_%H%M%S")
        else: period_to = sensor_data[6].strftime("%Y%m%d_%H%M%S")
        if sensor_id_from_history == sensor_id and customized_period_from >= period_from and customized_period_to <= period_to:
            print(sensor_data)
            sensor_location = sensor_data[1]
            machine_name = sensor_data[2]
            organization_sym = sensor_data[3]
            site_sym = sensor_data[4]

            cursor_xswh = conn_to_xswh.cursor()
            insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type, computation_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_balancing_mae, motor_balancing_machine_health, organization_sym, site_sym, file_name, 'balancing', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_misalignment_mae, motor_misalignment_machine_health, organization_sym, site_sym, file_name, 'misalignment', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_belt_mae, motor_belt_machine_health, organization_sym, site_sym, file_name, 'belt', 'model'))
            cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_bearing_mae, motor_bearing_machine_health, organization_sym, site_sym, file_name, 'bearing', 'model'))
            conn_to_xswh.commit()


preprocessing(fan_csv_paths_sorted)
preprocessing(motor_csv_paths_sorted)

fan_dataset_from_db = set()
motor_dataset_from_db = set()
fan_dataset_from_local = set()
motor_dataset_from_local = set()
temp_query = """
WITH cte AS (
     SELECT mae, health, computation_type, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp,
            TO_TIMESTAMP(SPLIT_PART(f.invoked_filename, '_', 2) || SPLIT_PART(f.invoked_filename, '_', 3), 'YYYYMMDDHH24MISS') AS extracted_datetime,
            ROW_NUMBER() OVER (PARTITION BY machine_name , health_type, sensor_location_name, invoked_filename ORDER BY invocation_timestamp DESC) AS rn
     FROM fact_machine_health f
     JOIN dim_sensor_info d ON f.sensor_info_id = d.id
     WHERE sensor_location_name IS NOT NULL AND health_type IS NOT NULL and machine_name in ('6B ISO Rm 1 Fan no.1', '6B ISO Rm 1 Fan no.2', '6B ISO Rm 2 Fan no.1', '6B ISO Rm 2 Fan no.2')
 )
 SELECT mae, health, computation_type, invoked_filename, node_id, sensor_location_name, machine_name, health_type, invocation_timestamp, extracted_datetime
 FROM cte
 WHERE rn = 1 and computation_type = 'model' and extracted_datetime >= '2023-10-18' and extracted_datetime <= '2023-11-18'
 ORDER BY invocation_timestamp DESC;
"""
cursor_xswh.execute(temp_query)
temp_query_result = cursor_xswh.fetchall()
for i in fan_csv_paths_sorted:
    file_name = os.path.basename(i)
    fan_dataset_from_local.add(file_name)
for i in temp_query_result:
    location_name = i[5]
    file_name = i[3]
    if location_name == 'Fan-DE':
        fan_dataset_from_db.add(file_name)

print('fan dataset from warehouse' ,len(fan_dataset_from_db))
print('fan dataset from local' ,len(fan_dataset_from_local))
# print('motor dataset from warehouse' ,len(motor_dataset_from_db))
count = 0
element = fan_dataset_from_local.difference(fan_dataset_from_db)
fan_dataset_from_local_sorted = sorted(element, key=lambda x: os.path.basename(x).split('_')[1:3])
for file in fan_dataset_from_local_sorted:
    count += 1
    print(f"{count} File is missing: {file}")
# docker run -it --rm --name my-running-app -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/out/models:/usr/src/app/out/models -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/data:/usr/src/app/data -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/src:/usr/src/app/src -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/src/test_model_copy.py:/usr/src/app/src/test_model_copy.py -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/.env:/usr/src/app/.env xs-tensorflow-app python src/test_model_copy.py

# SELECT period_from, period_to, location_name, m.machine_name, o.subdomain_name, site.site_id, node_id
# FROM sensor_history sh
# JOIN sensor_location sl ON sh.sensor_location_id = sl.id
# JOIN sensor s ON s.id = sh.sensor_id
# JOIN machine m ON sl.machine_id = m.id
# JOIN floorplan f ON f.id = m.floorplan_id
# JOIN site ON site.id = f.site_id
# JOIN organization o ON o.id = site.organization_id
# where site.site_id = 'tswh'
# order by period_to desc;