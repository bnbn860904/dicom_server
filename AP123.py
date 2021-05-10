from flask import Flask
from flask import jsonify, request
import numpy as np
from flask_cors import CORS
import os
import io
from flask import Flask, send_file

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    os.system('pvpython png2vti.py')
    return "Hello!"

@app.route('/upload', methods=['GET'])
def upload():
    with open("ESL.vti", 'rb') as bites:
        return send_file(
            io.BytesIO(bites.read()),
            mimetype='image/vti'
        )
    
   
    
if __name__ == '__main__':
    app.debug = False
    app.run(host='140.116.156.197', port=5000)    
    