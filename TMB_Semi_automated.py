import os 
import cv2 
import numpy as np
from pydicom import dcmread
import skimage
from skimage.morphology import disk
from skimage.morphology import binary_closing, binary_closing
from skimage.measure import label,regionprops
from skimage.filters import roberts
from scipy import ndimage
from scipy.interpolate import interpn

def bwperim(bw, n=4):
    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))
    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & (south == bw) & (west  == bw) & (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1]  = bw[1:, 1:]
        south_east[1:, 1:]    = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & (south_east == bw) & (south_west == bw) & (north_west == bw)
    return ~idx * bw

def signed_bwdist(im):
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    dist_im = ndimage.distance_transform_edt(1-im)
    return dist_im

def interp_shape(top, top_and_bottom, precision):
    r, c = top.shape
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    precision = 1+precision
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))    
    xi = np.c_[np.full((r*c),precision), xi]
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))
    out = out > 0
    return np.logical_not(out)

def get_dcm_img(dcm_path, WL, WW):
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
    
    t = WL + 0.5*WW
    b = WL - 0.5*WW
    minWindow = float(WL) - 0.5*float(WW)
    pixels_hu = (pixels_hu - minWindow) / float(WW)
    pixels_hu[pixels_hu < 0] = 0
    pixels_hu[pixels_hu > 1] = 1
    pixels_hu = (pixels_hu * 255).astype('uint8')
    return pixels_hu
    
def most_and_std(mask, hu):
    temp = label(mask)
    x_list = []
    y_list = []        
    for region in regionprops(temp):
        for coordinates in region.coords:
            x_list.append(coordinates[1])
            y_list.append(coordinates[0])     
    most = np.bincount(hu[y_list,x_list].astype(int)).argmax()
    std = np.std(hu[y_list,x_list])
    return most, std

def main(dcm_path, save_path, windowCenter, windowWidth, top_dcm, top_cont, middle_dcm, middle_cont, bottom_dcm, bottom_cont):
    name = os.listdir(dcm_path)
    idx_list = []
    for n in name:
        ds = dcmread(dcm_path + n)
        idx_list.append(int(ds.InstanceNumber))
    name = [name[i] for i in np.argsort(idx_list)]
    
    top_idx = name.index(top_dcm+'.dcm')
    mid_idx = name.index(middle_dcm+'.dcm')
    bottom_idx = name.index(bottom_dcm+'.dcm')
    print('top index, middle index, bottom index = ', top_idx, mid_idx, bottom_idx)    
    
    ds = dcmread(dcm_path+name[mid_idx])
    H = ds.pixel_array.shape[0]
    W = ds.pixel_array.shape[1]
    top_mask = np.zeros((H,W))
    cv2.fillPoly(top_mask, np.array([top_cont]), 255)
    middle_mask = np.zeros((H,W))
    cv2.fillPoly(middle_mask, np.array([middle_cont]), 255)
    bottom_mask = np.zeros((H,W))
    cv2.fillPoly(bottom_mask, np.array([bottom_cont]), 255)
    
    middle_hu = get_dcm_img(dcm_path+name[mid_idx], windowCenter, windowWidth) 
    temp_most, temp_std = most_and_std(middle_mask, middle_hu) 
    
    interp_top = signed_bwdist(np.logical_not(top_mask))
    interp_middle = signed_bwdist(np.logical_not(middle_mask))
    interp_bottom = signed_bwdist(np.logical_not(bottom_mask))
    interp_top_middle = np.stack((interp_top, interp_middle))
    interp_bottom_middle = np.stack((interp_bottom, interp_middle))
    
    seg_mask = middle_mask
    dcm_name_list = []
    contours_list = []
    binary_list = []
    
    for frontward in reversed(range(top_idx+1,mid_idx)):
        pixels_hu = get_dcm_img(dcm_path+name[frontward], windowCenter, windowWidth)
        pixels_hu = cv2.GaussianBlur(pixels_hu.astype('uint8'), (5, 5), 0)
        
        interp_mask = interp_shape(interp_top, interp_top_middle, (frontward-top_idx)/(mid_idx-top_idx))
        
        binary = np.zeros_like(pixels_hu)
        remain = (pixels_hu<=(temp_most+1.5*temp_std)) & (pixels_hu>=(temp_most-1.5*temp_std))
        binary[remain] = 255
        binary[seg_mask==0] = 0        
        
        keep_area = np.zeros_like(middle_mask)

        for r in regionprops(label(interp_mask)):                
            img = np.zeros_like(middle_mask)
            for c in r.coords:
                img[c[0], c[1]] = 1
            
            label_binary = label(binary)
            overlap_list = []                    
            for region in regionprops(label_binary):
                overlap = 0
                for coordinates in region.coords:
                    overlap = overlap + img[coordinates[0], coordinates[1]]
                overlap_list.append(overlap)
            i = 0
            for region in regionprops(label_binary):
                if i == np.argmax(overlap_list):
                    for coordinates in region.coords:
                        label_binary[coordinates[0], coordinates[1]] = -5  
                i = i+1
            keep_area[label_binary==-5] = 255

        binary[keep_area!=255] = 0
        binary = binary_closing(binary, disk(5))
        edges = roberts(binary)
        binary = ndimage.binary_fill_holes(edges)
        
        binary[seg_mask==0] = 0
        contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            dcm_name_list.append(name[frontward])
            temp = np.concatenate(([contours[i] for i in range(len(contours))]), axis=0)
            temp = np.squeeze(temp, axis=1)
            contours_list.append(temp)   
            
        seg_mask = binary 
        binary_list.insert(0,binary)
        
    seg_mask = middle_mask
    
    for backward in range(mid_idx+1,bottom_idx):
        pixels_hu = get_dcm_img(dcm_path+name[backward], windowCenter, windowWidth)
        pixels_hu = cv2.GaussianBlur(pixels_hu.astype('uint8'), (5, 5), 0)
        
        interp_mask = interp_shape(interp_bottom, interp_bottom_middle, (bottom_idx-backward)/(bottom_idx-mid_idx))  
        
        binary = np.zeros_like(pixels_hu)
        remain = (pixels_hu<=temp_most+1.5*temp_std) & (pixels_hu>=temp_most-1.5*temp_std)
        binary[remain] = 255
        binary[seg_mask==0] = 0
            
        keep_area = np.zeros_like(middle_mask)
        
        for r in regionprops(label(interp_mask)):
            img = np.zeros_like(middle_mask)
            for c in r.coords:
                img[c[0], c[1]] = 1

            label_binary = label(binary)
            overlap_list = []                    
            for region in regionprops(label_binary):
                overlap = 0
                for coordinates in region.coords:
                    overlap = overlap + img[coordinates[0], coordinates[1]]
                overlap_list.append(overlap)
            i = 0
            for region in regionprops(label_binary):
                if i == np.argmax(overlap_list):
                    for coordinates in region.coords:
                        label_binary[coordinates[0], coordinates[1]] = -5  
                i = i+1
            keep_area[label_binary==-5] = 255            

        binary[keep_area!=255] = 0
        binary = binary_closing(binary, disk(5))
        edges = roberts(binary)
        binary = ndimage.binary_fill_holes(edges)
        
        binary[seg_mask==0] = 0
        #cv2.imwrite(save_path + dcm_path.split('/')[-2]+'/'+name[backward][:-4]+'.png', binary.astype('uint8'))

        contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            dcm_name_list.append(name[backward])
            temp = np.concatenate(([contours[i] for i in range(len(contours))]), axis=0)
            temp = np.squeeze(temp, axis=1)
            contours_list.append(temp)  
            
        seg_mask = binary
        binary_list.append(binary)
    
    print('Save images.')
    num = 0
    for i in range(len(binary_list)-1):
        cv2.imwrite(save_path + dcm_path.split('/')[-2]+'/'+str(num)+'.png', (binary_list[i]*255).astype('uint8'))
        num = num + 1
        ds = dcmread(dcm_path+name[mid_idx])
        thickness = int(ds.SliceThickness)
        if thickness>1:
            for idx in range(1,thickness):
                t_ = signed_bwdist(np.logical_not(binary_list[i]))
                b_ = signed_bwdist(np.logical_not(binary_list[i+1]))
                t_b_ = np.stack((t_, b_))
                interp_mask = interp_shape(t_, t_b_, (idx/thickness))       
                cv2.imwrite(save_path + dcm_path.split('/')[-2]+'/'+str(num)+'.png', (interp_mask*255).astype('uint8'))
                num = num + 1
    print('Done.')
    return save_path + dcm_path.split('/')[-2]+'/'
