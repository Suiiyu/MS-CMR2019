import os
import cv2
import numpy as np
import nibabel as nib
import numpy as np
from skimage.transform import resize
import os, cv2
from sklearn import preprocessing as pre
import SimpleITK as sitk
from unet import unet
import shutil
import scipy
from keras import backend as K
from dataprocess import *

# Testing the 5 models.
VOLDATA_PATH = os.path.join("data", "lgedata")
WEIGHTS_PATH = "LOOmodel_logs/lge"
VOLMASK_PATH = os.path.join("data", "lgegt")
SAVE_PATH = "data/LOO/result"
INPUT_SHAPE = 144
weight = None
UNET_MODEL = None



def get_all_names(contour_path, shuffle=True):
    contours = os.listdir(contour_path)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    return contours


def load_image(path):
    img = nib.load(path)
    ref_affine = img.affine 
    origin_image_data = img.get_data().copy()
    return origin_image_data, ref_affine

    
def resize_image(resize_size, origin_image_data):
    height = origin_image_data.shape[-1]
    resized = np.zeros([resize_size[0], resize_size[1], height])
    for i in range(height):
        temp = origin_image_data[:, :, i] 
        resize_dim = list(resize_size) 
        compress_data = resize(temp, resize_dim, preserve_range=True)
        resized[:, :, i] = compress_data
    return resized 



def trans_to_validata_data(resized_image):
    data = np.zeros((resized_image.shape[2], input_size, input_size, 1))

    for i in range(resized_image.shape[2]):
        img = pre.minmax_scale(resized_image[:, :, i].astype('float32'))
        data[i, :, :, 0] = img
    return data

def center_crop(ndarray, crop_size):
    h, w = ndarray.shape
    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    cropped = ndarray[h_offset:(h_offset + crop_size),
              w_offset:(w_offset + crop_size)]
    return cropped


def padding(croped_img, uncroped_size): 
    w = uncroped_size
    h = uncroped_size
    crop_size = croped_img.shape[0]
    padding = np.zeros((input_size, input_size)).astype('float32')
    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    padding[h_offset:(h_offset + crop_size), w_offset:(w_offset + crop_size)] = croped_img
    return padding


def crop_for_input(batch, data):
    data_for_predict = np.zeros((batch, INPUT_SHAPE, INPUT_SHAPE, 1))
    for i in range(batch):
        data_for_predict[i, :, :, 0] = center_crop(data[i, :, :, 0], crop_size=INPUT_SHAPE)
    return data_for_predict



def unet_predict(weights, data, originshape, num_classes=4, input_size=256):
    batch = data.shape[0]  
    uncroped_size = data.shape[1] 
    data_for_predict = crop_for_input(batch, data) 
    UNET_MODEL.load_weights(weights, by_name=True)
    pred = UNET_MODEL.predict([data_for_predict], batch_size=16, verbose=1) 
    channel = pred.shape[-1]  

    predict = np.zeros((batch, uncroped_size, uncroped_size, channel)).astype("float32")
    predict_volume_data = np.zeros((originshape)).astype("int16")
    temp = np.zeros((originshape[0], originshape[1], channel)).astype("float32") 
    for i in range(batch): 
        for j in range(channel):
            predict[i, :, :, j] = padding(pred[i, :, :, j], uncroped_size=uncroped_size)
            resize_dim = [originshape[0], originshape[1]]
            temp[:, :, j] = resize(predict[i, :, :, j], list(resize_dim), preserve_range=True)
        predict_volume_data[:, :, i] = np.argmax(temp, axis=-1).astype("int16")
    return predict_volume_data


def merge_result(tempdata):
  
    one_hot_data_shape = np.eye(4)[tempdata[0]].shape
    temp = np.zeros(one_hot_data_shape).astype("int16")
    for data in tempdata:
        one_hot_data = np.eye(4)[data]
        temp = temp + one_hot_data
       
    result = np.argmax(temp, axis=-1).astype("int16") 
    result = change_label(result) 
    return result


def change_label(img):
    img[img == 1] = 200
    img[img == 2] = 500
    img[img == 3] = 600
    return img


def save_image(data, ref_affine, save_path):
    labeling_vol = nib.Nifti1Image(data, ref_affine)
    nib.save(labeling_vol, save_path) 


def remakedir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.mkdir(path)


def view_black_and_white(path, ctr, img, img_name):
    img[img == 200] = 80
    img[img == 500] = 160
    img[img == 600] = 240
    save_path = path
    if os.path.exists(save_path):
        pass
    else:
        remakedir(save_path)
    img_path = save_path + '/' + img_name + str(ctr) + ".png"

    scipy.misc.imsave(img_path, img)


def show(pre_path, gt_path, img_name):
    gt_data, _ = load_image( gt_path )
    for i in range(gt_data.shape[2]):
        view_black_and_white( "viewgt_fill", i,gt_data[:,:,i],img_name )
    pre_data, _ = load_image(pre_path)
    for i in range(pre_data.shape[2]):
        view_black_and_white("viewpre_fill", i, pre_data[:, :, i], img_name)


if __name__ == '__main__':
    input_size = 256 
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    val_list = ['patient1_LGE', 'patient2_LGE','patient3_LGE','patient4_LGE','patient5_LGE']
    vol_img_list = get_all_names(VOLDATA_PATH, shuffle=False)
    vol_gt_list = get_all_names(VOLMASK_PATH, shuffle=False)
    label_map=[200,500,600]
    for val_name in val_list:
        val = [val for _, val in enumerate(vol_img_list) if val.find(val_name)!=-1]
        val = val[0]
        val_gt = [val_gt for _, val_gt in enumerate(vol_gt_list) if val_gt.find(val_name) != -1]
        val_gt = val_gt[0]
        print('data:',val)
        print('mask:', val_gt)
        val_path = os.path.join(VOLDATA_PATH,val)
        val_data, ref_affine = load_image(val_path)

        val_gt, gt_affine = load_image(os.path.join(VOLMASK_PATH,val_gt))

        resized_image = resize_image([input_size, input_size], val_data)
        data = trans_to_validata_data(resized_image)
        weights_list = get_all_names(WEIGHTS_PATH, shuffle=False)
        weights = [weights for _,weights in enumerate(weights_list) if weights.find(val_name)!=-1]
        print('weights:',weights[0])
        UNET_MODEL = unet((INPUT_SHAPE, INPUT_SHAPE, 1), num_classes=4, lr=0, maxpool=True, weights=None)
        weight_path = os.path.join(WEIGHTS_PATH, weights[0])
        result = unet_predict(weight_path, data.copy(), val_data.shape)
        K.clear_session()
        result = change_label(result)
        rej_ratio = 0.3
        result = remove_minor_cc(result, rej_ratio=0.3)
        result = fillHole(result)
        path = os.path.join(SAVE_PATH, val)
        save_image(result, ref_affine, save_path=path)
        

        dices = []
        hds = []
        asds = []
        jaccards = []
        for i in label_map:
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            statistics_image_filter = sitk.StatisticsImageFilter()
            seg_map = (result==i)*1
            gt_map = (val_gt==i)*1
            seg_map = sitk.GetImageFromArray(seg_map)
            gt_map = sitk.GetImageFromArray(gt_map)
            seg_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg_map, squaredDistance=False))
            seg_surface = sitk.LabelContour(seg_map)
            gt_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_map, squaredDistance=False))
            gt_surface = sitk.LabelContour(gt_map)

           
            statistics_image_filter.Execute(gt_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())
       
            statistics_image_filter.Execute(seg_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            overlap_measures_filter.Execute(gt_map, seg_map)
            dice = overlap_measures_filter.GetDiceCoefficient()
            print('class: %d dice: %f'%(i,dice))
            jaccard = overlap_measures_filter.GetJaccardCoefficient()
            print('class: %d jaccard: %f' % (i, jaccard))

            hausdorff_distance_filter.Execute(gt_map, seg_map)
            hd = hausdorff_distance_filter.GetHausdorffDistance()
            print('class: %d hd: %f' % (i, hd))

            seg2ref_distance_map = gt_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
            ref2seg_distance_map = seg_distance_map * sitk.Cast(gt_surface, sitk.sitkFloat32)

            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            mref2seg = np.mean(ref2seg_distances)
            mseg2ref = np.mean(seg2ref_distances)
            msd = np.mean(all_surface_distances)
            print('class: %d msd: %f, mr2s: %f, ms2r: %f' % (i, msd,mref2seg,mseg2ref))
            hds.append(hd)
            dices.append(dice)
            asds.append(msd)
            jaccards.append(jaccard)

        print('data: %s, mean dice: %f, mean jaccard: %f, mean hd: %f, mean asd: %f' %(val,np.mean(dices),np.mean(jaccards),np.mean(hds),np.mean(asds)))


    print("Done!")







