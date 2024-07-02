extends Node

@onready var _server: WebSocketServer = get_node("../WebSocketServer")
@onready var _scene: Main = get_node("../");

signal game_start(difficulty);
signal game_stop;

var game_state = "i";
var game_score = "0";

var last_message = 0

var regex = RegEx.new()

func info(msg):
	print(msg)

# Server signals
func _on_web_socket_server_client_connected(peer_id):
	var peer: WebSocketPeer = _server.peers[peer_id]
	_server.send(peer_id, game_state + game_score)
	pass
#	info("Remote client connected: %d. Protocol: %s" % [peer_id, peer.get_selected_protocol()])
#	_server.send(-peer_id, "[%d] connected" % peer_id)


func _on_web_socket_server_client_disconnected(peer_id):
	var peer: WebSocketPeer = _server.peers[peer_id]
	pass
#	info("Remote client disconnected: %d. Code: %d, Reason: %s" % [peer_id, peer.get_close_code(), peer.get_close_reason()])
#	_server.send(-peer_id, "[%d] disconnected" % peer_id)

var last_1 = 0

func _on_web_socket_server_message_received(peer_id, message):
	if message == "":
		return
	if not regex.search(message):
		print(last_1)
		if message == "0":
			if last_1 > 0:
				_scene.ride(30000/last_1)
				last_1 = 0
			else:
				last_1 = 0
		else:
			last_1 += 1
	if message[0] == "r": # run
		game_start.emit(int(message[1]))
	if message[0] == "s": # stop
		game_stop.emit()
	pass
#	info("Server received data from peer %d: %s" % [peer_id, message])
#	_server.send(-peer_id, "[%d] Says: %s" % [peer_id, message])

func _set_game_state(game_state, game_score):
	self.game_state = str(game_state)
	self.game_score = str(game_score)
	_server.send(0, self.game_state + self.game_score)

# UI signals.
#func _on_send_pressed():
#	if _line_edit.text == "":
#		return
#
#	info("Sending message: %s" % [_line_edit.text])
#	_server.send(0, "Server says: %s" % _line_edit.text)
#	_line_edit.text = ""

func _ready():
	regex.compile("[^0-9]")
	
	var port = 9080
	var err = _server.listen(port)
	if err != OK:
		info("Error listing on port %s" % port)
		return
	info("Listing on port %s, supported protocols: %s" % [port, _server.supported_protocols])
