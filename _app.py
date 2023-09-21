from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
from learning import nn
# from visuallization import show_user_img
from test import minimalize

active_users = {}

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    user_id = request.sid  # 클라이언트의 세션 ID를 가져옵니다.
    active_users[user_id] = True

@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    active_users.pop(user_id, None)

@socketio.on('canvas_data')
def handle_canvas_data(canvas_data):
    session_id = canvas_data['sessionId']
    # print('Received canvas data:', canvas_data)  # 캔버스 데이터 출력
    canvas_data = np.array(canvas_data['canvas_data'])
    canvas_data = minimalize(canvas_data)
    canvas_data = ((canvas_data / 255.) - .5) * 2
    # show_user_img(canvas_data)
    result = int(nn.predict([canvas_data])[0])
    print(result)
    # print(canvas_data)
    
    socketio.emit('number', {'number': result}, room=session_id)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=80)
