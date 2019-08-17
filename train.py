import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
from keras.preprocessing.image import ImageDataGenerator
from unet import unet
from keras import backend as K
from dataprocess import *


seed = 1234
np.random.seed(seed)


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter))) ** power
    K.set_value(model.optimizer.lr, lrate)

    return K.eval(model.optimizer.lr)
def lr_ep_decay(model, base_lr, curr_ep, step=0.1):
    
    lrate = base_lr * step**(curr_ep/40)
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)
DATA_PATH = 'data/histmatch'
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train/img')
VAL_IMG_PATH=os.path.join(DATA_PATH, 'val/img')
TRAIN_MASK_PATH = os.path.join(DATA_PATH, 'train/gt')
VAL_MASK_PATH=os.path.join(DATA_PATH, 'val/gt')



def online_pre(model, online_ctrs, img, masks, state='test'):
    pred_img = model.predict([img], batch_size=32, verbose=1)
   
    final_img = np.argmax(pred_img,axis=-1)
    
    dices=get_all_dice(final_img,masks)
    acc=0
    for idx, ctr in enumerate(online_ctrs):
        tmp_pre = final_img[idx]
        tmp_img = np.argmax(masks[idx],axis=-1)
      
        temp=abs(tmp_pre-tmp_img)
        temp[temp > 0] = 1
        temp=temp.flatten()
        tempacc=1-np.sum(temp)/(final_img.shape[1]*final_img.shape[1])
        acc+=tempacc

        # if state == 'test':
        #     print('%s %f' % (ctr, mse))
        #     #test_res.write('%s %s\n'% (ctr, mse))
        #     #cv2.imwrite(os.path.join(onling_pre_save, '%s-pre.png' % (online_ctrs[idx])), tmp_pre*255)

    return acc/final_img.shape[0], dices


def get_dice(pred, mask):
    pred = pred.flatten()
    mask = mask.flatten()
    ints = 2.0*np.sum(pred*mask)
    sums = np.sum(pred + mask )
    dice = ints/(sums+0.0001)
    return dice


def get_all_dice(pred,mask):
    num_classes=4
    pred= np.eye( num_classes )[pred]
    dices=[]
    for i in range(1,4):
        print("predicting class ",i,"dice")
        temp=get_dice(pred[:,:,:,i],mask[:,:,:,i])
        dices.append(temp)
        print("predicting class:{} dice:{}".format(i,temp))
    return dices


if __name__ == '__main__':

    lr = 0.001
    input_size = 144
    
    input_shape = (input_size, input_size, 1)
    num_classes = 4
    
    weights = None
    
    model = unet(input_shape, num_classes, lr, maxpool=True, weights=weights)
    kwargs = dict(
        rotation_range=30,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 300
    mini_batch_size = 8
    
    train_ctrs = get_all_images(TRAIN_IMG_PATH, shuffle=True)#[:10]
                             

    
    dev_ctrs = get_all_images(VAL_IMG_PATH, shuffle=True)#[:10]
    dev_patients = ['patient1_LGE','patient2_LGE','patient3_LGE','patient4_LGE','patient5_LGE']
    for dev_test in dev_patients:
        print(dev_test)
        
        img_train = export_images(input_size,
                              train_ctrs,
                              TRAIN_IMG_PATH,
                              crop_size=input_size)
        mask_train = export_labels(input_size,
                              train_ctrs,
                              TRAIN_MASK_PATH,
                              crop_size=input_size) 
        print('1.mask train shape:{}'.format(mask_train.shape))
        
        test_res = open( 'logs/TestwCECropUNetlogLOO.txt', 'a+' )
        if not os.path.exists(dev_test + '_UNetmodel_logs'):
                os.makedirs(dev_test + '_UNetmodel_logs')
        test_ctrs = [x for _,x in enumerate(dev_ctrs) if x.find(dev_test)!=-1]
        dev_ctrs.remove(test_ctrs[0])
        img_dev = export_images(input_size,
                              dev_ctrs,
                              VAL_IMG_PATH,
                              crop_size=input_size)
        mask_dev = export_labels(input_size,
                              dev_ctrs,
                              VAL_MASK_PATH,
                              crop_size=input_size)

        print('mask dev shape:{}'.format(mask_dev.shape))
        #img_train = img_dev
        #mask_train = mask_dev
        
        for dev_index in range(len(img_dev)):
            img_train = np.append(img_train,[img_dev[dev_index]],axis=0)
            mask_train = np.append(mask_train,[mask_dev[dev_index]],axis=0)
        
        print('2.mask train shape:{}'.format(mask_train.shape))
        img_test = export_images(input_size,
                              test_ctrs,
                              VAL_IMG_PATH,
                              crop_size=input_size)
        mask_test = export_labels(input_size,
                              test_ctrs,
                              VAL_MASK_PATH,
                              crop_size=input_size)
        print('3.mask test shape:{}'.format(mask_test.shape))
    
        mask_train = np.eye(num_classes)[mask_train]
        mask_test = np.eye(num_classes)[mask_test]#one-hot
        print('4.mask train shape:{}'.format(mask_train.shape))
        
        image_generator = image_datagen.flow(img_train, shuffle=False,
                                         batch_size=mini_batch_size, seed=seed)

        mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                       batch_size=mini_batch_size, seed=seed)
        print('\nzip....')
        train_generator = zip(image_generator, mask_generator)

        max_iter = (len(train_ctrs) // mini_batch_size) * epochs

        step_iter = 1000
        curr_iter = 0
        print('curreniter')
        base_lr = K.eval(model.optimizer.lr)
        print('baseLi')
        curr_ep = 0
        lrate = lr_ep_decay(model, base_lr, curr_ep, step=0.5)
        print('\ntraining')
        # lrate = lr_step_decay(model, base_lr, curr_iter, step_iter, decay=0.1)


        for e in range(epochs):
            train_res = open('logs/TrainwCECropUNetlogLOO.txt', 'a+' )
            
            print('\nMain Epoch {:d}'.format(e + 1))
            print('\nLearning rate: {:6f}'.format(lrate))
            train_loss = []
            train_dice_list = []
            curr_ep = e+1
            if (e+1) % 40 == 0:
                lrate = lr_ep_decay(model, base_lr, curr_ep, step=0.5)
            for iteration in range(len(img_train) // mini_batch_size):
                img, mask = next(train_generator)
                res = model.train_on_batch([img], [mask])
                curr_iter += 1
                train_loss.append(res)

        
            train_loss = np.asarray(train_loss)
            train_loss = np.mean(train_loss, axis=0).round(decimals=10)
            print('Train result {}:\n{}'.format(model.metrics_names, train_loss))
            train_res.write('epoch:%s,loss:%s\n' % (e, train_loss))
        
            save_file = '_'.join(['MS_CMRUNetwCELGELOO',
                              'epoch', str(e + 1)]) + '.h5'
            
            save_path = os.path.join(dev_test + '_UNetmodel_logs', save_file)
            if (e+1) % 10 == 0:
           
                
                print('\n save path {}'.format(save_path))
                model.save_weights(save_path)
            train_res.close()
            
        test_acc, dices = online_pre(model,test_ctrs,img_test,mask_test,state='test')
        dicestr = ''
        for cls,dice in enumerate(dices):
            dicestr = str(cls+1) + ': ' + str(dice)+"---------" +dicestr
        dicestr = dicestr + "mean dice: "+ str(np.mean(dices))
        test_res.write('LOO_stage: %s, acc: %s, dice: %s\n' % (dev_test, str(test_acc), dicestr))
        test_res.close()
    K.clear_session()

