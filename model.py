import numpy as np

from tensorflow.keras.layers import (Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, 
                          Dropout, Conv2D, Conv2DTranspose, MaxPooling2D)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K


class SemiSupervisedVariatioanlAutoEncoder():
    def __init__(self, num_classes, num_samples, opts):
        #define model parameters
        self.labeled_vae_opt = opts[0]
        self.unlabeled_vae_opt = opts[1]
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.input_shape = (1,120,1)
        self.latent_dim = num_classes
        self.intermediate_dim = num_classes
        self.epsilon_std = 1.0
        self.dec_filters = 32
        self.enc_filters = 32
        self.cls_filters = 32
        self.enc_layers = [
            create_enc_conv_layers(stage=1, filters=self.dec_filters, kernel_size=(1, 5), strides=1),
            create_enc_conv_layers(stage=2, filters=self.dec_filters, kernel_size=(1, 5), strides=1),
            create_enc_conv_layers(stage=3, filters=self.dec_filters, kernel_size=(1, 5), strides=1),
            Flatten(),
            create_dense_layers(stage=4, width=self.intermediate_dim),] 
        self.classifier_layers = [
            Conv2D(filters=self.cls_filters, kernel_size=(1, 5)),
            Activation('relu'),
            Conv2D(filters=self.cls_filters, kernel_size=(1, 5)),
            Activation('relu'),
            Conv2D(filters=self.cls_filters, kernel_size=(1, 5)),
            Activation('relu'),
            Flatten(),
            Dense(self.num_classes),
            Activation('softmax'),    ]
        self.decoder_layers = [
            create_dense_layers(stage=10, width=self.dec_filters * 108 * 1),
            Reshape((1, 108, self.dec_filters)),
            create_dec_trans_conv_layers(stage=11, filters=self.dec_filters, kernel_size=(1, 5), strides=1),
            create_dec_trans_conv_layers(stage=12, filters=self.dec_filters, kernel_size=(1, 5), strides=1),
            create_dec_trans_conv_layers(stage=13, filters=self.dec_filters, kernel_size=(1, 5), strides=1),
            Conv2D(name='x_decoded', filters=1, kernel_size=5, padding='same', activation='sigmoid'),
            ]

        self.M2 = self.define_sVAE()

    def define_sVAE(self):
        x_in = Input(shape=self.input_shape,)
        y_in = Input(shape=(self.num_classes,))
        #define q(z|x)
        _enc_dense = inst_layers(self.enc_layers, x_in)
        # get mean and variance
        z_mean = Dense(self.latent_dim)(_enc_dense)
        z_log_var = Dense(self.latent_dim)(_enc_dense)
        # form the distribution
        def sampling(args, latent_dim = self.latent_dim, epsilon_std=self.epsilon_std):
            z_mean, z_log_var = args
        
            epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])


        # define q(y|x)
        _y_output = inst_layers(self.classifier_layers, x_in)
        # define q(x|z, y)
        # Labeled: 
        _merged = concatenate([y_in, z])
        _dec_out = inst_layers(self.decoder_layers, _merged)
        _x_output = _dec_out
        # Unlabeled:
        u_merged = concatenate([_y_output, z])
        u_x_output = inst_layers(self.decoder_layers, u_merged)
        #Loss Function
        def kl_loss(x, x_decoded_mean, z_mean=z_mean, z_log_var=z_log_var):
            kl_loss = - 0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
   
            return K.mean(kl_loss)

        def logxy_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = self.input_shape[1] * metrics.binary_crossentropy(x, x_decoded_mean)
            logy = np.log(1. / self.num_classes)
            
            return xent_loss - logy

        def labeled_vae_loss(x, x_decoded_mean):
            return logxy_loss(x, x_decoded_mean) + kl_loss(x, x_decoded_mean)

        def cls_loss(N):
            def cce_loss(y, y_pred):
                alpha = 0.1 * N
                return alpha * metrics.categorical_crossentropy(y, y_pred)
            return cce_loss

        def unlabeled_vae_loss(x, x_decoded_mean):
            entropy = metrics.categorical_crossentropy(_y_output, _y_output)
            labeled_loss = logxy_loss(x, x_decoded_mean) + kl_loss(x, x_decoded_mean) 
            return K.mean(K.sum(_y_output * labeled_loss, axis=-1)) + entropy

        #Compile Model
        labeled_vae = Model(inputs=[x_in, y_in], outputs=[_x_output, _y_output])
        labeled_vae.compile(optimizer=self.labeled_vae_opt, loss=[labeled_vae_loss, cls_loss(self.num_samples)])
        labeled_vae.summary()
        unlabeled_vae = Model(inputs=x_in, outputs=u_x_output)
        unlabeled_vae.compile(optimizer=self.unlabeled_vae_opt, loss=unlabeled_vae_loss)
        unlabeled_vae.summary()
        classifier = Model(inputs=[x_in], outputs=[_y_output])
        classifier.compile(optimizer=self.labeled_vae_opt, loss=cls_loss(self.num_samples))
        return labeled_vae, unlabeled_vae, classifier

def create_enc_conv_layers(stage, **kwargs):
    conv_name = '_'.join(['enc_conv', str(stage)])
    bn_name = '_'.join(['enc_bn', str(stage)])
    layers = [
        Conv2D(name=conv_name, **kwargs),
        #BatchNormalization(name=bn_name),
        Activation('relu'),
    ]
    return layers

def create_dense_layers(stage, width):
    dense_name = '_'.join(['enc_conv', str(stage)])
    bn_name = '_'.join(['enc_bn', str(stage)])
    layers = [
        Dense(width, name=dense_name),
        #BatchNormalization(name=bn_name),
        Activation('relu'),
        Dropout(0.2),
    ]
    return layers

def create_dec_trans_conv_layers(stage, **kwargs):
    conv_name = '_'.join(['dec_trans_conv', str(stage)])
    bn_name = '_'.join(['dec_bn', str(stage)])
    layers = [
        Conv2DTranspose(name=conv_name, **kwargs),
        #BatchNormalization(name=bn_name),
        Activation('relu'),
    ]
    return layers

def inst_layers(layers, in_layer):
    x = in_layer
    for layer in layers:
        if isinstance(layer, list):
            x = inst_layers(layer, x)
        else:
            x = layer(x)
        
    return x



