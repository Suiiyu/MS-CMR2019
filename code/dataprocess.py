import os
import cv2
import numpy as np
from sklearn import preprocessing as pre

def read_data(contour,img_path,label_path,crop_size=None):
    # read images and labels
    img_full_path = os.path.join(img_path, contour)
    img = cv2.imread(img_full_path, -1).astype('float32')
    #img = pre.minmax_scale(img)
    label_full_path = os.path.join(label_path, contour)
    label = cv2.imread(label_full_path, -1).astype('float32')
    #label = pre.minmax_scale(label)
    if crop_size!=None:
        img = center_crop(img,crop_size)
        label = center_crop(label,crop_size)
    
    
    return img, label
        
        
def read_img(contour,img_path,crop_size=None):
    #just read images
    img_full_path = os.path.join(img_path, contour)
    img = cv2.imread(img_full_path, -1).astype('float32')
    img = pre.minmax_scale(img)
    if crop_size!=None:
        img = center_crop(img,crop_size)
    return img
    
    
def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    '''
    h, w= ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')

    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)
        pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w, d = ndarray.shape
    # center crop
    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    cropped = ndarray[h_offset:(h_offset + crop_size),
              w_offset:(w_offset + crop_size)]

    return cropped
    
def get_all_images(contour_path, shuffle=True):
    contours = os.listdir(contour_path)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    return contours
    
    
def export_images(input_size, contours,data_path, crop_size=None):
    images = np.zeros((len(contours), input_size, input_size, 1))
    for idx, contour in enumerate(contours):
        img = read_img(contour,data_path,crop_size)
        images[idx,:,:,0] = img
    return images
    
def export_labels(input_size, contours,mask_path,crop_size=None):
    labels = np.zeros((len(contours), input_size, input_size)).astype('int16')
    for idx, contour in enumerate(contours):
        label = read_label(contour,mask_path,crop_size)
        labels[idx] = label.astype('int16')
    return labels    
def export_all_contours(input_size, contours, data_path, mask_path, crop_size=None):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))

    images = np.zeros((len(contours), input_size, input_size, 1))
    
    masks = np.zeros((len(contours), input_size, input_size,1))
    for idx, contour in enumerate(contours):
        #print('name:',contour)
        img, mask = read_data(contour, data_path, mask_path, crop_size=crop_size)
        
        images[idx,:,:,0] = img
        masks[idx,:,:,0] = mask
        
    return images, masks
    
    
def read_label(contour,label_path, crop_size=None):

    label_full_path = os.path.join(label_path, contour)
    # print(img_full_path,mask_full_path)
    label = cv2.imread(label_full_path, -1)
    # print(mask.shape)
    if crop_size!=None:
        label = center_crop(label,crop_size)


    return  label
def remove_minor_cc(vol_data, rej_ratio, rename_map=[0,200,500,600]):
    """Remove small connected components refer to rejection ratio"""

    rem_vol = copy.deepcopy(vol_data)
    class_n = len(rename_map)
    # retrieve all classes
    for c in range(1, class_n):
        print ('processing class %d...' % c)


        class_idx = (vol_data==rename_map[c])*1
        class_vol = np.sum(class_idx)
        labeled_cc, num_cc = measurements.label(class_idx)
        # retrieve all connected components in this class
        for cc in range(1, num_cc+1):
            single_cc = ((labeled_cc==cc)*1)
            single_vol = np.sum(single_cc)
            # remove if too small
            if single_vol / (class_vol*1.0) < rej_ratio:
                # print('remove small connected from class:%d' % c)
                rem_vol[labeled_cc==cc] = 0
       
    return rem_vol

def fillHole(im_in,class_level=[0,200,500,600]):

    fill_volume = copy.deepcopy(im_in)

    h, w, z = im_in.shape

    for slice_num in range(z):
        img = copy.deepcopy(im_in[:,:,slice_num])
        for level in range(1,len(class_level)):
        # Floodfill from point (0, 0)
            img_idx = (im_in[:,:,slice_num]!=class_level[level])*1
            img_num = np.sum(img_idx)
            labeled_cc, num_cc = measurements.label(img_idx)
            for cc in range(1, num_cc + 1):
                single_cc = ((labeled_cc == cc) * 1)
                single_vol = np.sum(single_cc)
                # remove if too small
                if single_vol / (img_num * 1.0) < 0.0005:
                    img[labeled_cc == cc] = class_level[level]
        fill_volume[:,:,slice_num] = img

    return fill_volume
