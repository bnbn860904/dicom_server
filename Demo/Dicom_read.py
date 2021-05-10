### Get CT images from DICOM ###
 
import numpy as np
from pydicom import dcmread

def get_dcm_img(dcm_path):
    ds = dcmread(dcm_path)
    def get_pixels_hu(slices):
        image = slices.pixel_array
        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = slices.RescaleIntercept  #Intercept
        slope = slices.RescaleSlope  #Rescale
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)
    pixels_hu = get_pixels_hu(ds)
        
    temp = pixels_hu
    temp = temp[temp>=90]
    temp = temp[temp<=110]
    counts = np.bincount(temp)
    
    pixels_hu[pixels_hu<-50] = -50
    pixels_hu[pixels_hu>200] = 200

    minimum = np.min(pixels_hu)
    maximum = np.max(pixels_hu)
    pixels_hu = (pixels_hu-minimum)*255./(maximum-minimum)
    
    pixels_hu = pixels_hu.astype(np.float32)
    return pixels_hu, np.sum(counts)