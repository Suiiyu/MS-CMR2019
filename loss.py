import numpy as np
import tensorflow as tf

def softmax_weighted_loss():
    def compute_loss(gt, y_pred):
        n_dims=y_pred.shape[-1]
        loss = 0.
        #############################################
        # gt = produce_mask_background( gt, y_pred)  
        # gt = tf.stop_gradient( gt )
        #############################################
        for i in range(n_dims):
            gti = gt[:,:,:,i]
            predi = y_pred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
            focal_loss=1
            loss = loss + -tf.reduce_mean(weighted * gti * focal_loss * tf.log(tf.clip_by_value(predi, 0.005, 1 )))
        return loss
    return compute_loss
def compute_softmax_weighted_loss(gt, y_pred):
    n_dims=y_pred.shape[-1]
    loss = 0.
    #############################################
    # gt = produce_mask_background( gt, y_pred)  
    # gt = tf.stop_gradient( gt )
    #############################################
    for i in range(n_dims):
        gti = gt[:,:,:,i]
        predi = y_pred[:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
        focal_loss=1
        loss = loss + -tf.reduce_mean(weighted * gti * focal_loss * tf.log(tf.clip_by_value(predi, 0.005, 1 )))
    return loss

#
# class softmax_weighted_loss():
#     """
#     local (over window) normalized cross correlation
#     """
#
#     def __init__(self, win=None, eps=1e-5):
#         self.win = win
#         self.eps = eps
#
#     def ncc(self, I, J):
#         
#         print(I.shape)
#         print( J.shape )
#         # get dimension of volume
#         # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
#         ndims = len(I.get_shape().as_list()) - 2
#         assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
#
#         # set window size
#         if self.win is None:
#             self.win = [9] * ndims
#
#         # get convolution function
#         conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
#
#         # compute CC squares
#         I2 = I * I
#         J2 = J * J
#         IJ = I * J
#
#         # compute filters
#         sum_filt = tf.ones([*self.win, 1, 1])
#         strides = [1] * (ndims + 2)
#         padding = 'SAME'
#
#         # compute local sums via convolution
#         I_sum = conv_fn(I, sum_filt, strides, padding)
#         J_sum = conv_fn(J, sum_filt, strides, padding)
#         I2_sum = conv_fn(I2, sum_filt, strides, padding)
#         J2_sum = conv_fn(J2, sum_filt, strides, padding)
#         IJ_sum = conv_fn(IJ, sum_filt, strides, padding)
#
#         # compute cross correlation
#         win_size = np.prod(self.win)
#         u_I = I_sum / win_size
#         u_J = J_sum / win_size
#
#         cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
#         I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
#         J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
#
#         cc = cross * cross / (I_var * J_var + self.eps)
#
#         # return negative cc.
#         return tf.reduce_mean(cc)
#
#     def loss(self, I, J):
#         return - self.ncc(I, J)
def get_loss():

    right_loss=softmax_weighted_loss()
    left_loss =softmax_weighted_loss() 
    loss_metric=[right_loss,left_loss]
    
    loss_weight_metric=[1,0.5]
    
    return loss_metric,loss_weight_metric

def background_num_to_save(input_gt,pred):
    background_num = tf.reduce_sum(input_gt[ :, :, :,0])
    total_num=tf.reduce_sum(input_gt)
    foreground_num=total_num-background_num
    save_back_ground_num=tf.reduce_max([2*foreground_num, background_num/8])
    save_back_ground_num=tf.clip_by_value(save_back_ground_num, 0, background_num)
    return save_back_ground_num

def no_background(input_gt):
    return input_gt

def exist_background(input_gt, pred,save_back_ground_num):
   
    in_width= pred.get_shape()[1]
    in_height=in_width
    pred_data = pred[:, :,  :, 0]  
    gt_backgound_data=1-input_gt[:, :, :, 0]
    pred_back_ground_data = tf.reshape(pred_data, (-1,  in_height * in_width)) 
    gt_back_ground_data=tf.reshape(gt_backgound_data, (-1,  in_height * in_width))
    new_pred_data=pred_back_ground_data+gt_back_ground_data

    gti = -1*new_pred_data[:, :]
    
    max_k_number, index = tf.nn.top_k(gti, save_back_ground_num)
    
    max_k = tf.reduce_min(max_k_number) 
    one = tf.ones_like(gti)
    zero = tf.zeros_like(gti) 
    mask_slice = tf.where(gti < max_k, x=zero, y=one) 
    mask_slice = tf.reshape(mask_slice, (-1,  in_height, in_width))
    temp=input_gt[:, :, :, 0] * mask_slice
    temp1 = tf.expand_dims( temp, -1 )
    temp2=input_gt[:, :, :, 1]
    temp2 = tf.expand_dims( temp2, -1 )
    temp3 = input_gt[:, :, :, 2]
    temp3 = tf.expand_dims( temp3, -1 )
    temp4 = input_gt[:, :, :, 3]
    temp4 = tf.expand_dims( temp4, -1 )
   
    input_gt=tf.concat( [temp1, temp2,temp3,temp4], axis=3 )
    return input_gt


def produce_mask_background(input_gt,pred):
    save_back_ground_num=background_num_to_save(input_gt,pred)
    save_back_ground_num = tf.cast(save_back_ground_num, dtype=tf.int32)
    product = tf.cond(save_back_ground_num < 5, lambda:no_background(input_gt),lambda: exist_background(input_gt, pred,save_back_ground_num))
    
    return product
