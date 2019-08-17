from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda, Input, average, Reshape, UpSampling2D, Multiply,Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, GlobalAveragePooling2D, Flatten, Dense, Add, \
    AveragePooling2D, Conv1D
from keras.layers import ZeroPadding2D, Cropping2D, BatchNormalization, MaxPooling2D
from keras import backend as K
from keras import losses
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import regularizers
from loss import compute_softmax_weighted_loss

# m: input
# dim: the num of channel
# res: controls the res connection
# drop: controls the dropout layer
# initpara: initial parameters
def convblock(m, dim, layername, res=1, drop=0, **kwargs):
    n = Conv2D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    n = Dropout(drop)(n) if drop else n
    n = Conv2D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    return Concatenate()([m, n]) if res else n

def unet(input_shape, num_classes, lr, maxpool=True, weights=None):
    '''initialization'''
    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform', 
        #kernel_initializer='he_normal',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True)

    num_classes = num_classes
    data = Input(shape=input_shape, dtype='float', name='data')
    # encoder
    enconv1 = convblock(data, dim=32, layername='block1', **kwargs)
    pool1 = MaxPooling2D(pool_size=3, strides=2,padding='same',name='pool1')(enconv1) if maxpool \
        else Conv2D(filters=32, strides=2, name='pool1')(enconv1)

    enconv2 = convblock(pool1, dim=64, layername='block2', **kwargs)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool2')(enconv2) if maxpool \
        else Conv2D(filters=64, strides=2, name='pool2')(enconv2)

    enconv3 = convblock(pool2, dim=128, layername='block3', **kwargs)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool3')(enconv3) if maxpool \
        else Conv2D( filters=128, strides=2, name='pool3')(enconv3)

    enconv4 = convblock(pool3, dim=256, layername='block4', **kwargs)
    pool4 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool4')(enconv4) if maxpool \
        else Conv2D(filters=256, strides=2, name='pool4')(enconv4)

    enconv5 = convblock(pool4, dim=512, layername='block5notl', **kwargs)
    # decoder
    up1 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu',
                 name='up1')(UpSampling2D(size=[2, 2])(enconv5))
    merge1 = Concatenate()([up1,enconv4])
    deconv6 = convblock(merge1, dim=256, layername='deconv6', **kwargs)

    up2 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
                 name='up2')(UpSampling2D(size=[2,2])(deconv6))
    merge2 = Concatenate()([up2,enconv3])
    deconv7 = convblock(merge2, dim=128, layername='deconv7', **kwargs)

    up3 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
                 name='up3')(UpSampling2D(size=[2, 2])(deconv7))
    merge3 = Concatenate()([up3, enconv2])
    deconv8 = convblock(merge3, dim=64, layername='deconv8', **kwargs)

    up4 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',
                 name='up4')(UpSampling2D(size=[2, 2])(deconv8))
    merge4 = Concatenate()([up4, enconv1])
    deconv9 = convblock(merge4, dim=32, drop=0.5, layername='deconv9', **kwargs)
    conv10 = Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu',
                     name='conv10')(deconv9)

    predictions = Conv2D(filters=num_classes, kernel_size=1, activation='softmax',
                          padding='same', name='predictions')(conv10)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights,by_name=True)
    sgd = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=sgd, loss=compute_softmax_weighted_loss, metrics=['accuracy'])
    #model.summary()
    return model


if __name__ == '__main__':
    model = unet((256, 256, 1), 1, 0.001, maxpool=True, weights=None)

