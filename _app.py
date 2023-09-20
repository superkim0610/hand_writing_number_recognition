from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from learning import nn
# from visuallization import show_user_img

import random

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('canvas_data')
def handle_canvas_data(canvas_data):
    print('Received canvas data:', canvas_data)  # 캔버스 데이터 출력
    canvas_data = np.array(canvas_data)
    canvas_data = ((canvas_data / 255.) - .5) * 2
    # show_user_img(canvas_data)
    result = int(nn.predict([canvas_data])[0])
    # print(canvas_data)
    
    socketio.emit('number', {'number': result})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=80)
