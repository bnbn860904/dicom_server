from flask import Flask, send_file
from flask import jsonify, request
import singleCT_test
import numpy as np
from flask_cors import CORS
import os
import io
import CV
import TMB_Semi_automated

app = Flask(__name__)
CORS(app)

coordinate_top = 0
patient_top = 0



@app.route("/")
def hello():
    return "Hello!"

       
@app.route('/patient', methods=['GET']) ## tumor_segmentation(AI)
def patient_seg():
    results = []
    
    if 'number' in request.args:
        number = request.args['number']
        number = number + ".dcm"

    else:
        print("error")

##-------------------main_function_1------------------##    
    
    liver_contours, tumor_contours = singleCT_test.main(number)
    if len(liver_contours) == 0:
        liver_list = []
    else:
        liver_list = (liver_contours[0]).tolist()
        #liver_list = liver_list[0][0]
        A_list = []
        k = len(liver_list)
        for i in range(0, k): 
            A_list += liver_list[i]
        
    if len(tumor_contours) == 0:
        tumor_list = []
    else:
        tumor_list = (tumor_contours[0]).tolist()
        
    patient = {
        
    "liver":A_list,
    "tumor":tumor_list
        
     }
    
    
    
    
    return jsonify(patient)

##-------------------main_function_1------------------## 
        
@app.route('/tumorCV', methods=['GET'])  ## tumor_segmentation(CV)
def CV_seg():
    results = []
    
    str_url = ""
    
    if 'number' in request.args:
        number = request.args['number']
        str_url = number.split('-')
        patient_id = str_url[0]
        coordinate = int(str_url[1]),int(str_url[2])
        

    else:
        print("error")
   
##-------------------main_function_2------------------##    
    
    cv_contours = CV.region_growing_API(patient_id,coordinate)
    
    cv_contours = (cv_contours).tolist()
    B_list = []
    k = len(cv_contours)
    for i in range(0, k):     
        B_list += cv_contours[i]        
    
    """if len(cv_contours) == 0:
        coo_result = []
    else:
        A_list = []
        h = len(cv_contours)
        for t in range(0, h):
            list1 = (cv_contours[0]).tolist()                    
            k = len(list1)
            for i in range(0, k): 
                A_list += list1[i]
    """   
        
    patient2 = {
        
    #"patient_id":patient_id,
    "coordinate":B_list
        
     }
       
    return jsonify(patient2)

##-------------------main_function_2------------------## 

##-------------------main_function_3_top------------------##
@app.route('/3D_top', methods=['GET'])  ## 3D_top_segmentation
def top():
    results = []
    
    str_url = ""
    
    global coordinate_top
    
    if 'number' in request.args:
        number = request.args['number']
        str_url = number.split('-')
        coordinate_top = str_url
        
        
    if 'patient_id' in request.args:
        patient_id = request.args['patient_id']
        
        patient_top = patient_id
    mytest = {
    
     }
       
    return jsonify(mytest)
##-------------------main_function_3_top------------------##

##-------------------main_function_4_middle------------------##
@app.route('/3D_middle', methods=['GET'])  ## 3D_middle_segmentation
def middle():
    results = []
    
    str_url = ""
    
    global coordinate_mid
    global WL
    
    if 'number' in request.args:
        number = request.args['number']
        str_url = number.split('-')
        coordinate_mid = str_url
        
        #print(str_url)
        
    if 'patient_id' in request.args:
        patient_id = request.args['patient_id']
        #print(patient_id)
        
    if 'WL' in request.args:
        WL_list = request.args['WL']
        WL = WL_list.split('-')
        
        
    mytest = {
    
     }
       
    return jsonify(mytest)
##-------------------main_function_4_middle------------------##

##-------------------main_function_5_bottom------------------##
@app.route('/3D_bottom', methods=['GET'])  ## 3D_bottom_segmentation
def bottom():
    results = []
    
    str_url = ""
    
    global coordinate_bot
    
    if 'number' in request.args:
        number = request.args['number']
        str_url = number.split('-')
        coordinate_bot = str_url
        
    if 'patient_id' in request.args:
        patient_id = request.args['patient_id']
        #print(patient_id)
        
    mytest = {
    
     }
       
    return jsonify(mytest)
##-------------------main_function_5_bottom------------------##
##-------------------main_function_6_start_drawing------------------##
@app.route('/start_drawing', methods=['GET'])  ## start_drawing
def start():
        
    top = xy_array(coordinate_top)
    print(top)
    mid = xy_array(coordinate_mid)
    print(mid)
    bot = xy_array(coordinate_bot)
    print(bot)
    
    print(WL)
    L = int(WL[0])
    W = int(WL[1])
    
    input_file = TMB_Semi_automated.main('C:/Users/leowang/Desktop/00A00101/', 'C:/Users/leowang/Desktop/test/',L, W, '00A00101_3_07', top, '00A00101_3_13', mid, '00A00101_3_30', bot)
    com_line = 'pvpython png2vti.py --input ' + input_file + ' --output volume99'
    os.system(com_line)
    
    mytest = {
    
     }
       
    return jsonify(mytest)
##-------------------main_function_6_start_drawing------------------##

@app.route('/upload', methods=['GET'])
def upload():
    with open("C:/Users/leowang/Desktop/volume99.vti", 'rb') as bites:
        return send_file(
            io.BytesIO(bites.read()),
            mimetype='image/vti'
        )
##-------------------main_function_upload---------------------------##
def xy_array(a_list):
    A = []
    k = len(a_list)
    for i in range(0, k-2,2):
        #print(i)
        A.append([int(a_list[i]),int(a_list[i+1])])
                
    return A
    
    
    
    
    
if __name__ == '__main__':
    app.debug = False
    app.run(host='140.116.156.197', port=5000)    
    