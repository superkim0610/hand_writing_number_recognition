from flask import Flask, jsonify, render_template
from subprocess import call
from flask_socketio import SocketIO, send

app = Flask(__name__)
socket_io = SocketIO(app)


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
    print('received message : ', message)
    socket_io.emit('message', message)

if __name__ == '__main__':
    app.run()