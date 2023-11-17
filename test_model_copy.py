import os
import tensorflow as tf
import pandas as pd
import numpy as np
from models.low_vel_gan import LowVelGAN
import glob
from dotenv import load_dotenv
import psycopg2

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

#Define desired patterns for filtering the unwanted files
allowed_fan_sensor_ids = ['189249773', '189265662', '189265907', '189265924', '189265970', '189286743', '189286744', '189286750', '189286774', '189286790', '189286793', '189286804', '189286837']
allowed_motor_sensor_ids = ['189249000', '189249352', '189257988', '189265943', '189270035', '189286742', '189286752', '189286761', '189286763', '189286799', '189286801', '189286831', '189286845']
data_path = 'data/vel/s3'
# Model path
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
fan_record_count = motor_record_count = 0
# Loop through all the dataset and do the prediction
for machine_type in ['fan', 'motor']:
# for machine_type in ['motor', 'fan']:
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
    machine_sensor_ids = allowed_fan_sensor_ids if machine_type == 'fan' else allowed_motor_sensor_ids
    for sensor_id in machine_sensor_ids:
        sensor_path = f"{data_path}/{machine_type}/{sensor_id}/*.csv"
        file_list = glob.glob(sensor_path)
        for file_path in file_list:
            try:
                # print("file_path: ", file_path)
                invoked_filename = file_path.split("/")[-1]
                print("invoked_filename: ", invoked_filename)
                # Check if the invoked filename already exists in the database
                if machine_type == 'fan':
                    check_existed_query = "SELECT COUNT(DISTINCT health_type) FROM staging_machine_health WHERE invoked_filename = %s AND health_type IN ('balancing', 'misalignment', 'belt', 'bearing', 'flow')"
                    cursor_xswh.execute(check_existed_query, (invoked_filename,))
                    fan_record_count = cursor_xswh.fetchone()[0]
                elif machine_type == 'motor':
                    check_existed_query = "SELECT COUNT(DISTINCT health_type) FROM staging_machine_health WHERE invoked_filename = %s AND health_type IN ('balancing', 'misalignment', 'belt', 'bearing')"
                    cursor_xswh.execute(check_existed_query, (invoked_filename,))
                    motor_record_count = cursor_xswh.fetchone()[0]
                    
                # If no record matched:
                print("fan_record_count:", fan_record_count)
                print("motor_record_count:", motor_record_count)
                if fan_record_count != 5 and motor_record_count != 4:
                    data = pd.read_csv(file_path)
                    data = data.drop(columns=[data.columns[-1]])
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
                    bearing_actual_data = np.reshape(bearing_input_data, (1,2048,8,1) )
                    if machine_type == 'fan':
                        # Fan model prediction
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
                        fan_balancing_mae = np.mean(np.absolute(np.subtract(balancing_actual_data, fan_balancing_prediction_result)))
                        fan_misalignment_mae = np.mean(np.absolute(np.subtract(misalignment_actual_data, fan_misalignment_prediction_result)))
                        fan_belt_mae = np.mean(np.absolute(np.subtract(belt_actual_data, fan_belt_prediction_result)))
                        fan_flow_mae = np.mean(np.absolute(np.subtract(flow_actual_data, fan_flow_prediction_result)))
                        fan_bearing_mae = np.mean(np.absolute(np.subtract(bearing_actual_data, fan_bearing_prediction_result)))
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
                        # XS database query
                        cursor_xsdb = conn_to_xsdb.cursor()
                        # Select the sensor_location and machine_name from xs db
                        select_xsdb_query = """select sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor s
                        join sensor_location sl on sl.sensor_id = s.id
                        join machine m on sl.machine_id = m.id
                        join floorplan f on f.id = m.floorplan_id
                        join site on site.id = f.site_id
                        join organization o on o.id = site.organization_id
                        where s.node_id = %s"""
                        cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))
                        # Get the data back from query
                        selected_row = cursor_xsdb.fetchone()
                        sensor_location = selected_row[0] if selected_row else None
                        machine_name = selected_row[1] if selected_row else None
                        organization_sym = selected_row[2] if selected_row else None
                        site_sym = selected_row[3] if selected_row else None
                        # Close the connection of the xsdb
                        cursor_xsdb.close()
                        # XS warehouse query
                        cursor_xswh = conn_to_xswh.cursor()
                        insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_balancing_mae, fan_balancing_machine_health, organization_sym, site_sym, invoked_filename, 'balancing'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_misalignment_mae, fan_misalignment_machine_health, organization_sym, site_sym, invoked_filename, 'misalignment'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_belt_mae, fan_belt_machine_health, organization_sym, site_sym, invoked_filename, 'belt'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_flow_mae, fan_flow_machine_health, organization_sym, site_sym, invoked_filename, 'flow'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, fan_bearing_mae, fan_bearing_machine_health, organization_sym, site_sym, invoked_filename, 'bearing'))
                        conn_to_xswh.commit()
                    elif machine_type == 'motor':
                        # Motor model prediction
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
                        cursor_xsdb = conn_to_xsdb.cursor()
                        # Select the sensor_location and machine_name from xs db
                        select_xsdb_query = """select sl.location_name as sensor_location, m.machine_name, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor s
                        join sensor_location sl on sl.sensor_id = s.id
                        join machine m on sl.machine_id = m.id
                        join floorplan f on f.id = m.floorplan_id
                        join site on site.id = f.site_id
                        join organization o on o.id = site.organization_id
                        where s.node_id = %s"""
                        cursor_xsdb.execute(select_xsdb_query, (int(sensor_id),))
                        # Get the data back from query
                        selected_row = cursor_xsdb.fetchone()
                        sensor_location = selected_row[0] if selected_row else None
                        machine_name = selected_row[1] if selected_row else None
                        organization_sym = selected_row[2] if selected_row else None
                        site_sym = selected_row[3] if selected_row else None
                        # Close the connection of the xsdb
                        cursor_xsdb.close()
                        # # XS warehouse query
                        cursor_xswh = conn_to_xswh.cursor()
                        insert_xswh_query = "INSERT INTO staging_machine_health (node_id, sensor_location_name, machine_name, mae, health, organization_sym, site_sym, invoked_filename, health_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_balancing_mae, motor_balancing_machine_health, organization_sym, site_sym, invoked_filename, 'balancing'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_misalignment_mae, motor_misalignment_machine_health, organization_sym, site_sym, invoked_filename, 'misalignment'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_belt_mae, motor_belt_machine_health, organization_sym, site_sym, invoked_filename, 'belt'))
                        cursor_xswh.execute(insert_xswh_query, (int(sensor_id), sensor_location, machine_name, motor_bearing_mae, motor_bearing_machine_health, organization_sym, site_sym, invoked_filename, 'bearing'))
                        conn_to_xswh.commit()
                # If already existed, skip insertion
                else:
                    print("Invoked filename already exists in the database. Skipping insertion.")
            except:
                print("Empty File")
    # Close the connection of the xswh
    conn_to_xsdb.commit()
    conn_to_xsdb.close()
    cursor_xswh.close()
    conn_to_xswh.close()

# Script: docker run -it --rm --name my-running-app -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/out/models:/usr/src/app/out/models -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/data/vel/s3:/usr/src/app/data/vel/s3 -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/src:/usr/src/app/src -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/src/test_model_copy.py:/usr/src/app/src/test_model_copy.py xs-tensorflow-app python src/test_model_copy.py

# Querying the desired data with in the range 2023-01-01 to 2023-10-31 (both include)
"""
select * from fact_machine_health f join dim_sensor_info d on f.sensor_info_id = d.id where split_part(invoked_filename, '_', 2) <= '20231031' and split_part(invoked_filename, '_', 2) >= '20230101' order by split_part(invoked_filename, '_', 2) desc;
"""
"""
select * from fact_machine_health f join dim_sensor_info d on f.sensor_info_id = d.id where split_part(invoked_filename, '_', 2) <= '20231031' and split_part(invoked_filename, '_', 2) >= '20230101' and d.old_health_type is null order by invocation_timestamp desc;
"""

'''
SELECT *
FROM staging_raw_machine_health
WHERE machine_id IN (
  '5A ISO Rm 1 Fan no.2',
  '5A ISO Rm 2 Fan no.1',
  '6A ISO Rm 1 fan no.1',
  '6A ISO Rm 1 Fan no.2',
  '6A ISO Rm 2 Fan no.1',
  '6A ISO Rm 2 Fan no.2',
  '6B ISO Rm 1 Fan no.1',
  '6B ISO Rm 1 Fan no.2',
  '6B ISO Rm 2 Fan no.1',
  '6B ISO Rm 2 Fan no.2',
  '7A ISO Rm 1 Fan no.1',
  '7A ISO Rm 2 Fan no.2'
)
ORDER BY last_updated desc;
'''
# Select sensor history
"""
select sl.location_name, machine_name, node_id, period_from, period_to, o.subdomain_name as organization_sym, site.site_id as site_sym from sensor_history sh join sensor_location sl on sh.sensor_location_id = sl.id join sensor s on s.id = sh.sensor_id join machine m on sl.machine_id = m.id join floorplan f on f.id = m.floorplan_id join site on site.id = f.site_id join organization o on o.id = site.organization_id where site.site_id = 'tswh' order by period_from desc;
"""

# Select how many sensors are bound with machine.
"""
SELECT machine_name, s.node_id, period_from, period_to, location_name
FROM sensor_history sh
JOIN sensor_location sl ON sh.sensor_location_id = sl.id
JOIN sensor s ON s.id = sh.sensor_id
JOIN machine m ON sl.machine_id = m.id
JOIN floorplan f ON f.id = m.floorplan_id
JOIN site ON site.id = f.site_id
JOIN organization o ON o.id = site.organization_id
WHERE site.site_id = 'tswh'
GROUP BY machine_name, s.node_id, period_from, period_to, location_name
HAVING machine_name = '6B ISO Rm 1 Fan no.2'
order by period_from desc;
"""

"""
SELECT machine_name, s.node_id, period_from, period_to, location_name
FROM sensor_history sh
JOIN sensor_location sl ON sh.sensor_location_id = sl.id
JOIN sensor s ON s.id = sh.sensor_id
JOIN machine m ON sl.machine_id = m.id
JOIN floorplan f ON f.id = m.floorplan_id
JOIN site ON site.id = f.site_id
JOIN organization o ON o.id = site.organization_id
WHERE site.site_id = 'tswh'
GROUP BY machine_name, s.node_id, period_from, period_to, location_name
order by period_from desc;
"""