from flask import Flask
from flask import jsonify, request
import singleCT_test
import numpy as np
from flask_cors import CORS

import CV

app = Flask(__name__)
CORS(app)


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
    
if __name__ == '__main__':
    app.debug = False
    app.run(host='140.116.156.197', port=5000)    
    