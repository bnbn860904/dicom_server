### convert json to label png ###
 
import numpy as np
import cv2
import json
import os
import os.path as osp

def test(data):
    classes_name = ['background', 'Liver','Nodules']    
    cls_map = {name: i for i, name in enumerate(classes_name)}
    fill_color = [0,127,255]
    height = data['imageHeight']
    width = data['imageWidth']
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:] = fill_color[0]
    ### draw Liver first, then draw Nodules
    for shape in data['shapes']:
        if shape['label'] == 'Liver':
            points = shape['points']
            cv2.fillPoly(mask, np.array([points], dtype=np.int32), fill_color[1])
    for shape in data['shapes']:
        if shape['label'] == 'Nodules':
            points = shape['points']
            cv2.fillPoly(mask, np.array([points], dtype=np.int32), fill_color[2])
    return mask
def main(path, outpath):    
    for file in (os.listdir(path)):
        path_ = path + file + '/'       
        outpath_ = outpath + file + '/' 
        if not osp.exists(outpath_):
            os.mkdir(outpath_)
        ##### create dataset #####  
        for i in (os.listdir(path_)):            
            if i.endswith('.dcm'):
                ##### create label image #####
                json_path = path_+i[:-3]+'json'
                if os.path.exists(json_path):                    
                    data = json.load(open(json_path))
                    mask = test(data)                    
                    cv2.imwrite(outpath_ + '//' + i[:-3] + 'png', mask)
                else:
                    lbl = np.zeros((512, 512), dtype=np.uint8)
                    cv2.imwrite(outpath_ + '//' + i[:-3] + 'png', lbl)
        print('Saved to: %s' % file)
if __name__ == '__main__':
    #path = 'C:/Users/harris/Desktop/耀瑄科技計畫/liver tumor/'
    path = './Train data/Dicom file/'
    #outpath = 'C:/Users/harris/Desktop/99999999999999/'
    outpath = './Train data/Label file/'
    if not osp.exists(outpath):
        os.mkdir(outpath)
    main(path, outpath)