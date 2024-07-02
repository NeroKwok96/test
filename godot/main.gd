extends Node3D

class_name Main

signal _set_game_state(game_state, game_score)

@onready var bike = $"./BikePath/BikeFollow/Bike"
@onready var enemy_bike = $"./EnemyPath/BikeFollow/Bike"
@onready var enemy2_bike = $"./EnemyPath2/BikeFollow/Bike"
@onready var camera_progress = $"./BikePath/BikeFollow/Bike/Path3D/PathFollow3D"
@onready var bike_animation = $"./BikePath/BikeFollow/Bike/bike/AnimationPlayer"
@onready var bike_woman_animation = $"./BikePath/BikeFollow/Bike/exercising-woman/AnimationPlayer"
@onready var bike_follow: PathFollow3D = $"./BikePath/BikeFollow/"
@onready var bike_path: Path3D = $"./BikePath/"
@onready var enemy_animation = $"./EnemyPath/BikeFollow/Bike/bike/AnimationPlayer"
@onready var enemy_woman_animation = $"./EnemyPath/BikeFollow/Bike/exercising-woman/AnimationPlayer"
@onready var enemy_follow: PathFollow3D = $"./EnemyPath/BikeFollow/"
@onready var enemy2_animation = $"./EnemyPath2/BikeFollow/Bike/bike/AnimationPlayer"
@onready var enemy2_woman_animation = $"./EnemyPath2/BikeFollow/Bike/exercising-woman/AnimationPlayer"
@onready var enemy2_follow: PathFollow3D = $"./EnemyPath2/BikeFollow/"

@onready var speed_label = $"./Speed"
@onready var speedmeter_label = $"./SpeedMeter"
@onready var points_label = $"./Points"
@onready var time_label = $"./Time"
@onready var countdown_label = $"./Countdown"
# Test
@onready var lap_label = $"./Lap"

@onready var timer = $"./Timer"

var started = false
var finished = false
var countdown = false	
var speed = 0
var acceleration = 0
var enemy_speed = 0
var score = 0
var time = 60
var difficulty = 0
var total_length = 0
var player_progress = 0
var enemy_progress = 0
var enemy2_progress = 0
var record_player_progress = 0
var record_enemy_progress = 0
var record_enemy2_progress = 0

# Test
var final_lap = 4

# Called when the node enters the scene tree for the first time.
func _ready():
	bike_woman_animation.current_animation = "Riding Mountain Bike"
	bike_woman_animation.pause()
	bike_animation.current_animation = "Bike Animation"
	bike_animation.pause()
	enemy_woman_animation.current_animation = "Riding Mountain Bike"
	enemy_woman_animation.pause()
	enemy_animation.current_animation = "Bike Animation"
	enemy_animation.pause()
	
	# Test
	enemy2_woman_animation.current_animation = "Riding Mountain Bike"
	enemy2_woman_animation.pause()
	enemy2_animation.current_animation = "Bike Animation"
	enemy2_animation.pause()
	
	_set_game_state.emit("i", score)
	total_length = bike_path.curve.get_baked_length()

func _game_start(_difficulty):
	started = true
	finished = false
	countdown = true
	score = 0
	player_progress = 0
	enemy_progress = 0
	enemy2_progress = 0
	bike_follow.progress = 0
	enemy_follow.progress = 0
	enemy2_follow.progress = 0
	enemy_bike.position[1] = 0.09 # Y position should be 0.09
	enemy2_bike.position[1] = 0.25 # Y position should be 0.25
	camera_progress.progress_ratio = 0.03

	speed = 0
	enemy_speed = 0
	difficulty = _difficulty
	timer.start(4)
	update_status()
	pass

func _game_end():
	finished = true
	record_player_progress = player_progress
	record_enemy_progress = enemy_progress
	timer.stop()
	update_status()
	
func _game_stop():
	started = false
	finished = false
	score = 0
	update_status()
	pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	var diff_x = delta * speed / 3
	var diff_animation = delta * speed / 50
	
	var enemy_diff_x = delta * enemy_speed / 3
	var enemy_diff_animation = delta * enemy_speed / 50
	
	var enemy2_diff_x = delta * enemy_speed / 3
	var enemy_dff_animation = delta * enemy_speed / 50
	
	# bike.translate(Vector3(-diff_x, 0, 0))
	enemy_animation.advance(enemy_diff_animation)
	enemy_woman_animation.advance(enemy_diff_animation)
	enemy_progress += enemy_diff_x
	enemy_follow.progress = enemy_progress

	enemy2_animation.advance(enemy_diff_animation)
	enemy2_woman_animation.advance(enemy_diff_animation)
	enemy2_progress += enemy_diff_x
	enemy2_follow.progress = enemy_progress
	
	bike_animation.advance(diff_animation)
	bike_woman_animation.advance(diff_animation)
	player_progress += diff_x
	bike_follow.progress = player_progress
	
#	speed_label.text = str(round(score)) + "m"
	speedmeter_label.text = str(round(speed / 2)) + "km/h"
	var lap = max(1, ceil(player_progress/total_length))
	var enemy_lap = max(1, ceil(enemy_progress/total_length))
	var enemy2_lap = max(1, ceil(enemy2_progress/total_length))
	
	## Test
	if lap >= 2 and lap < final_lap:
		lap_label.text = "Lap " + str(lap)
		if snapped(player_progress/total_length, 0.01) > lap - 1 + 0.10:
			lap_label.text = ""
			
	elif lap == final_lap:
		lap_label.text = "Final Lap"
		if snapped(player_progress/total_length, 0.01) > lap - 1 + 0.10:
			lap_label.text = ""
	
	# For enemy
	if (((enemy_progress / total_length) > enemy_lap - 1 + 0.10 and (enemy_progress / total_length) <= enemy_lap - 1 + 0.24)
	or ((enemy_progress / total_length) > enemy_lap - 1 + 0.60 and (enemy_progress / total_length) <= enemy_lap - 1 + 0.74)):
		enemy_bike.translate(Vector3(0,0.0010,0))
	elif (((enemy_progress / total_length) > enemy_lap - 1 + 0.24 and (enemy_progress / total_length) <= enemy_lap - 1 + 0.38)
	or ((enemy_progress / total_length) > enemy_lap - 1 + 0.74 and (enemy_progress / total_length) <= enemy_lap - 1 + 0.88)):
		enemy_bike.translate(Vector3(0,-0.0010,0))
	
	# For enemy 2
	if (((enemy2_progress / total_length) > enemy2_lap - 1 + 0.10 and (enemy2_progress / total_length) <= enemy2_lap - 1 + 0.24)
	or ((enemy2_progress / total_length) > enemy2_lap - 1 + 0.60 and (enemy2_progress / total_length) <= enemy2_lap - 1 + 0.74)):
		enemy2_bike.translate(Vector3(0,0.0020,0))
	elif (((enemy2_progress / total_length) > enemy2_lap - 1 + 0.24 and (enemy2_progress / total_length) <= enemy2_lap - 1 + 0.38)
	or ((enemy2_progress / total_length) > enemy2_lap - 1 + 0.74 and (enemy2_progress / total_length) <= enemy2_lap - 1 + 0.88)):
		enemy2_bike.translate(Vector3(0,-0.0020,0))
	
	if lap <= final_lap:
		points_label.text = "Laps: " + str(lap) + "/" + str(final_lap)
	
	if finished:
		if record_enemy_progress > record_player_progress:
			speed_label.text = "1. Computer\n2. You (-" + str(floor(record_enemy_progress - record_player_progress)) + "m)"
		else:
			speed_label.text = "1. You (+" + str(floor(record_player_progress - record_enemy_progress)) + "m)\n2. Computer"
	else:
		if enemy_progress > player_progress:
			speed_label.text = "1. Computer\n2. You (-" + str(floor(enemy_progress - player_progress)) + "m)"
		else:
			speed_label.text = "1. You (+" + str(floor(player_progress - enemy_progress)) + "m)\n2. Computer"
	
	if countdown:
		if int(timer.time_left) > 0:
			time_label.text = ""
			countdown_label.text = str(int(timer.time_left))
		else:
			time_label.text = ""
			countdown_label.text = "Go!"
	else:
		time_label.text = str(int(timer.time_left))
		if finished:
			countdown_label.text = "FINISHED"
		else:
			countdown_label.text = ""

	if started and not countdown:
		speed = min(speed + acceleration * delta, 250)
		enemy_speed = enemy_speed + 70 * delta * min(difficulty, 2.5) * max(1, speed/100)
		enemy_speed = max(100, enemy_speed * (1 - 0.4 * delta))
		enemy_speed = min(enemy_speed, 150 * difficulty + speed / 7)
		#print(enemy_speed)
	
	speed = max(0, speed * (1 - 0.4 * delta))
	acceleration = acceleration * 0.5
	print(acceleration)
	if countdown == false and finished == false:
		score += speed/10 * delta 
	
	if countdown:
		camera_progress.progress_ratio = min(1, camera_progress.progress_ratio + 0.5 * delta)

	if not finished:
		if lap > final_lap:
			_game_end()
		if enemy_lap > final_lap:
			_game_end()
		if enemy2_lap > final_lap:
			_game_end()

func _tick():
	update_status()
	if int(round(timer.time_left)) == 0:
		if countdown:
			timer.start(60)
			countdown = false
		else:
			_game_end()
func _input(ev):
	if Input.is_key_pressed(KEY_S):
		#_game_start(1)
		_game_start(1.35)
	if Input.is_key_pressed(KEY_D):
		#_game_start(1.2)
		_game_start(1.365)
	if Input.is_key_pressed(KEY_F):
		_game_start(1.38)
	if Input.is_key_pressed(KEY_M):
		ride(4000)

func ride(power):
	if started and not countdown and not finished:
		acceleration = power / 4
	update_status()

func update_status():
	if started:
		if finished:
			_set_game_state.emit("f", round(score))
		else:
			_set_game_state.emit("r", round(score))
	else:
		_set_game_state.emit("i", round(score))

func stop():
	pass
