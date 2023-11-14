import pandas as pd
import numpy as np
import tensorflow as tf
# from models.low_vel_gan import LowVelGAN

#Define desired patterns for filtering the unwanted files
allowed_fan_sensor_ids = ['189249773', '189265662', '189265907', '189265924', '189265970', '189286743', '189286744', '189286750', '189286774', '189286790', '189286793', '189286804', '189286837']
#Define desired patterns for filtering the unwanted files
allowed_motor_sensor_ids = ['189249000', '189249352', '189257988', '189265943', '189270035', '189286742', '189286752', '189286761', '189286763', '189286799', '189286801', '189286831', '189286845']
desired_pattern = 'vel_freq'

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

fan_data_path = "189249773_20230101_011247_vel_freq.csv"
fan_data = pd.read_csv(fan_data_path)
fan_data = fan_data.drop(columns=[fan_data.columns[-1]])
# Fan balancing
fan_balancing_df = fan_data.to_numpy()[:128, 1:3]
fan_balancing_squared_values = np.square(fan_balancing_df)
fan_balancing_input_data = np.concatenate([fan_balancing_df, fan_balancing_squared_values], axis=-1)
fan_balancing_actual_data = np.reshape(fan_balancing_input_data, (1,128,4,1))
# Fan misalignment
fan_misalignment_df = fan_data.to_numpy()[99:323, 3:]
fan_misalignment_squared_values = np.square(fan_misalignment_df)
fan_misalignment_input_data = np.concatenate([fan_misalignment_df, fan_misalignment_squared_values], axis=-1)
fan_misalignment_actual_data = np.reshape(fan_misalignment_input_data, (1,224,2,1))
# Fan belt
fan_belt_df = fan_data.to_numpy()[:128, 1:]
fan_belt_squared_values = np.square(fan_belt_df[:, 0:2])
fan_belt_vertical_variance = np.abs(fan_belt_df[:, 0:1] - np.max(fan_belt_df[:, 0:1]))
fan_belt_horizontal_variance = np.abs(fan_belt_df[:, 1:2] - np.max(fan_belt_df[:, 1:2]))
fan_belt_axial_variance = np.abs(fan_belt_df[:, 2:] - np.max(fan_belt_df[:, 2:]))
fan_belt_input_data = np.concatenate([fan_belt_df, fan_belt_squared_values, fan_belt_vertical_variance, fan_belt_horizontal_variance, fan_belt_axial_variance], axis=-1)
fan_belt_actual_data = np.reshape(fan_belt_input_data, (1,128,8,1))
# Fan flow
fan_flow_df_1 = fan_data.to_numpy()[884:884 + 42, 1:3]
fan_flow_df_2 = fan_data.to_numpy()[1780:1780 + 43, 1:3]
fan_flow_df_3 = fan_data.to_numpy()[2676:2676 + 43, 1:3]
fan_flow_input_data = np.concatenate([fan_flow_df_1, fan_flow_df_2, fan_flow_df_3], axis=0)
fan_flow_actual_data = np.reshape(fan_flow_input_data, (1,128,2,1))
# Fan bearing
fan_bearing_df = fan_data.to_numpy()[:2048, 1:]
fan_bearing_squared_values = np.square(fan_bearing_df[:, 0:2])
fan_bearing_vertical_variance = np.abs(fan_bearing_df[:, 0:1] - np.sqrt(np.sum(np.square(fan_belt_df[:, 0:1])) / 1.5))
fan_bearing_horizontal_variance = np.abs(fan_bearing_df[:, 1:2] - np.sqrt(np.sum(np.square(fan_belt_df[:, 1:2])) / 1.5))
fan_bearing_axial_variance = np.abs(fan_bearing_df[:, 2:] - np.sqrt(np.sum(np.square(fan_belt_df[:, 2:])) / 1.5))
fan_bearing_input_data = np.concatenate([fan_bearing_df, fan_bearing_squared_values, fan_bearing_vertical_variance, fan_bearing_horizontal_variance, fan_bearing_axial_variance], axis=-1)
# # Model Prediction
# fan_balancing_model = LowVelGAN(input_shape=fan_balancing_actual_data.shape)
# fan_balancing_model.load_keras_models(save_dir=fan_balancing_model_path)
# fan_balancing_prediction_result = fan_balancing_model.generator.predict(fan_balancing_actual_data)[1]
# fan_misalignment_model = LowVelGAN(input_shape=fan_misalignment_actual_data.shape)
# fan_misalignment_model.load_keras_models(save_dir=fan_misalignment_model_path)
# fan_misalignment_prediction_result = fan_misalignment_model.generator.predict(fan_misalignment_actual_data)[1]
# fan_belt_model = LowVelGAN(input_shape=fan_belt_actual_data.shape)
# fan_belt_model.load_keras_models(save_dir=fan_belt_model_path)
# fan_belt_prediction_result = fan_belt_model.generator.predict(fan_belt_actual_data)[1]
# fan_flow_model = LowVelGAN(input_shape=fan_flow_actual_data.shape)
# fan_flow_model.load_keras_models(save_dir=fan_flow_model_path)
# fan_flow_prediction_result = fan_flow_model.generator.predict(fan_flow_actual_data)[1]
# # Mae
# fan_balancing_mae = np.mean(np.absolute(np.subtract(fan_balancing_actual_data, fan_balancing_prediction_result)))
# fan_misalignment_mae = np.mean(np.absolute(np.subtract(fan_misalignment_actual_data, fan_misalignment_prediction_result)))
# fan_belt_mae = np.mean(np.absolute(np.subtract(fan_belt_actual_data, fan_belt_prediction_result)))
# fan_flow_mae = np.mean(np.absolute(np.subtract(fan_flow_actual_data, fan_flow_prediction_result)))
# print('fan_balancing_mae: ', fan_balancing_mae)
# print('fan_misalignment_mae: ', fan_misalignment_mae)
# print('fan_belt_mae: ', fan_belt_mae)
# print('fan_flow_mae: ', fan_flow_mae)
# fan_balancing_machine_health = calculate_machine_health(fan_balancing_mae, fan_balancing_threshold[0], fan_balancing_threshold[1], fan_balancing_threshold[2])

# Script: docker run -it --rm --name my-running-app -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/out/models:/usr/src/app/out/models -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/data/vel/s3:/usr/src/app/data/vel/s3 -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/src:/usr/src/app/src -v C:/Users/hikar/Documents/Code/xs-tswh-vel-spec-ai/src/test_model_copy.py:/usr/src/app/src/test_model_copy.py xs-tensorflow-app python src/test_model_copy.py