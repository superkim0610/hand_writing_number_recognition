from flask import Flask, jsonify, render_template, request
from subprocess import call
from flask_socketio import SocketIO, send

app = Flask(__name__)
socket_io = SocketIO(app)

active_users = {}

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/json_test')
def json_test():
    data = {'result': 2}
    return jsonify(data)

@app.route('/socket_test')
def socket_test():
    return render_template('socket_test.html')

@socket_io.on('message')
def handle_message(message):
    user_id = request.sid()
    active_users[user_id] = True
    socket_io.emit('message', {'data': 'connected!'}, room=user_id)
    print('received message : ', message)
    print('sid : ', user_id)

if __name__ == '__main__':
    socket_io.run(app, debug=True, host='0.0.0.0', port=80)