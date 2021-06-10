import time
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler 
from matplotlib import pyplot as plt


from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from model import *
from utils import *

def fit_model(labeled_vae, unlabeled_vae, X_unlabeled, X_labeled, y_labeled, epochs, batch_size):
    start = time.time()
    for epoch in range(epochs):
        unlabeled_index = np.arange(len(X_unlabeled)) #produce index with the same amount of len(X_unlabeled)
        np.random.shuffle(unlabeled_index) # shuffle index
        # Repeat the labeled data to match length of unlabeled data
        labeled_index = []
        for i in range(len(X_unlabeled) // len(X_labeled)):
            l = np.arange(len(X_labeled))
            np.random.shuffle(l)
            labeled_index.append(l)

        labeled_index = np.concatenate(labeled_index)
        
        batches = len(X_unlabeled) // batch_size
        for i in range(batches):
            # Labeled
            index_range =  labeled_index[i * batch_size:(i+1) * batch_size]
            loss = labeled_vae.train_on_batch([X_labeled[index_range], y_labeled[index_range]], 
                                            [X_labeled[index_range], y_labeled[index_range]])
            
            # Unlabeled
            index_range =  unlabeled_index[i * batch_size:(i+1) * batch_size]
            loss += [unlabeled_vae.train_on_batch(X_unlabeled[index_range],  X_unlabeled[index_range])]
            print('Epoch: %s/%s Loss: %s' %(epoch+1, epochs, loss))


def run_exp():
    # experiment setup #
    dataset = data_preproc(np.asarray(pickle.load(open('dataset/EXP1.pickle','rb'))))
    n_classes = 16
    n_samples_list = [16] #10752
    batch_size = 180
    epochs = 100
    run_times = 1
    optimizers = [Adam(lr=0.0001, beta_1=0.5), Adam(lr=0.0001, beta_1=0.5)]
    X_train, y_train, X_tst, y_tst = dataset
    print(X_train.shape,  y_train.shape, X_tst.shape, y_tst.shape)

    # run exps under differnt numbers of labeled samples:
    for j in range(len(n_samples_list)):
        history = []
        print('Fitting with sample_size: {}'.format(n_samples_list[j]))

        #selected sample for semisupervised
        if n_samples_list[j] < len(X_train):
            sss = StratifiedShuffleSplit(n_splits=2, test_size=n_samples_list[j] / len(X_train), random_state=0)
            _, index = sss.split(X_train, y_train)
            X, y = X_train[index[1]], y_train[index[1]]
            X_others, _ = X_train[index[0]], y_train[index[0]]
        else:
            X, y = X_train, y_train
            X_others = X_train
        y = to_categorical(y)

        for i in range(run_times):
            labeled_vae, unlabeled_vae, classifier = SemiSupervisedVariatioanlAutoEncoder(n_classes, n_samples_list[j], optimizers).M2
            fit_model(labeled_vae, unlabeled_vae, X_others, X, y, epochs, batch_size) ## apply X_others = X if want fully-lableled
            y_pred = np.argmax(classifier.predict(X_tst), axis=-1)
            score = accuracy_score(y_tst, y_pred)
            print('ACC:', score)
            history.append(score)
        best = max (history) # adopt best acc after run_times


        #fh = open('exp3/SVAE-{}-{}.pickle'.format(n_samples_list[j], best),'wb')
        #pickle.dump(history, fh)
        #fh.close()

        # generate reconstructed CSI
        #def generate_samples(samples, vae, n_samples):
        #    generated_samples = vae.predict(samples)
        #    print(generated_samples.shape)
        #    fh = open('dataset/reconstructed/SVAE-X-{}.pickle'.format(n_samples),'wb')
        #    pickle.dump(generated_samples, fh)
        #    fh.close()
        #generate_samples(X_train, unlabeled_vae, n_samples_list[j])




def run_exp3():
    # experiment setup 
    dataset_r1 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r1.pickle','rb'))))
    dataset_r2 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r2.pickle','rb'))))
    n_classes = 18
    n_samples_list = [18] #10752
    batch_size = 180
    epochs = 100
    run_times = 1
    optimizers = [Adam(lr=0.0001, beta_1=0.5), Adam(lr=0.0001, beta_1=0.5)]
    X_train1, y_train1, X_tst1, y_tst1 = dataset_r1
    X_train2, y_train2, X_tst2, y_tst2 = dataset_r2

    # run exps under differnt numbers of labeled samples:
    for j in range(len(n_samples_list)):
        history = []
        print('Fitting with sample_size: {}'.format(n_samples_list[j]))

        #selected sample for semisupervised
        if n_samples_list[j] < len(X_train1):
            sss = StratifiedShuffleSplit(n_splits=2, test_size=n_samples_list[j] / len(X_train1), random_state=0)
            _, index = sss.split(X_train1, y_train1)
            X1, y1 = X_train1[index[1]], y_train1[index[1]]
            X_others1, _ = X_train1[index[0]], y_train1[index[0]]
        else:
            X1, y1 = X_train1, y_train1
            X_others1 = X_train1

        if n_samples_list[j] < len(X_train2):
            sss = StratifiedShuffleSplit(n_splits=2, test_size=n_samples_list[j] / len(X_train2), random_state=0)
            _, index = sss.split(X_train2, y_train2)
            X2, y2 = X_train2[index[1]], y_train2[index[1]]
            X_others2, _ = X_train2[index[0]], y_train2[index[0]]
        else:
            X2, y2 = X_train2, y_train2
            X_others2 = X_train2


        X = np.concatenate((X1, X2))
        y = np.concatenate((y1, y2))
        X_tst = np.concatenate((X_tst1, X_tst2))
        y_tst = np.concatenate((y_tst1, y_tst2))

        y = to_categorical(y)
        X_others = np.concatenate((X_others1, X_others2))

        for i in range(run_times):
            seed(i)
            tf.set_random_seed(i)
            labeled_vae, unlabeled_vae, classifier = SemiSupervisedVariatioanlAutoEncoder(n_classes, n_samples_list[j], optimizers).M2
            #train(labeled_vae, unlabeled_vae, X_train, X_sup, y_sup, epochs, batch_size)
            fit_model(labeled_vae, unlabeled_vae, X_others, X, y, epochs, batch_size)
            y_pred = np.argmax(classifier.predict(X_tst), axis=-1)
            score = accuracy_score(y_tst, y_pred)
            print('ACC:', score)
            history.append(score)

        best = max (history) # adopt best acc after run_times

        # generate reconstructed CSI
        #generated_samples1 = unlabeled_vae.predict(X_train1)
        #generated_samples2 = unlabeled_vae.predict(X_train2)
        #fh = open('dataset/reconstructed/SVAE-r1-X-{}.pickle'.format(n_samples_list[j]),'wb')
        #pickle.dump(generated_samples1, fh)
        #fh.close()
        #fh = open('dataset/reconstructed/SVAE-r2-X-{}.pickle'.format(n_samples_list[j]),'wb')
        #pickle.dump(generated_samples2, fh)
        #fh.close()

        #fh = open('exp3/SVAEr1r2-{}-{}.pickle'.format(n_samples_list[j], best),'wb')
        #pickle.dump(history, fh)
        #fh.close()


if __name__ == '__main__':

    seed(0)
    run_exp()


    '''
    exp1: 17, 0
    exp2: 14, 0
    exp3: 1, 2 
    '''


