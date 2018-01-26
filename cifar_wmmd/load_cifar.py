
import cPickle
import numpy as np
import random
import pdb
from scipy.misc import imresize
from scipy.misc import imsave

random.seed(1) # set a seed so that the results are consistent


def load_data(split='train'):
    """ Defines the two-class dataset and fetches from source.

    All data is resized from 32x32 to 28x28, and returned in NCHW format,
    with values corresponding to pixel intensity, i.e. {0, 1, ..., 255}.

    Potential cifar10 pairings:
      Frog - 6, Ship - 8 

    Potential cifar100 pairings:
      Apple - 0, Orange - 53
      Computer keyboard - 39, Willow tree - 96

    Args:
      split: Keyword to identify 'train' vs 'test'.
    
    Returns:
      images_0: Numpy array of images from the first class.
      images_1: Numpy array of images from the second class.
      labels_0: Numpy array of labels from the first class.
      labels_1: Numpy array of labels from the second class.
           
    """
    dataset = 'cifar10'  # ['cifar10', 'cifar100']

    if dataset == 'cifar100':
        path = 'data/cifar-100-python/'
        label_0 = 39
        label_1 = 96

        f = open(path + split, 'rb')
        dict = cPickle.load(f)

        # Load only two classes, and code as binary one-hot labels.
        fine_labels = np.array(dict['fine_labels'])
        indices_0 = np.where(fine_labels == 39)[0] 
        indices_1 = np.where(fine_labels == 96)[0] 

        # Fetch images, resize to 28x28, and flatten again.
        images_0 = np.array([dict['data'][i] for i in indices_0])  # (data_size, 3072)
        images_1 = np.array([dict['data'][i] for i in indices_1])

    elif dataset == 'cifar10':
        path = 'data/cifar-10-batches-py/'
        label_0 = 6
        label_1 = 8

        if split == 'train':
            batch_names = (
                ['data_batch_1', 'data_batch_2', 'data_batch_3',
                 'data_batch_4', 'data_batch_5'])

            all_ind_0 = np.array([])
            all_ind_1 = np.array([]) 
            all_images = np.zeros(shape=[50000, 3072]) 

            for i, batch in enumerate(batch_names):
                f = open(path + batch, 'rb')
                dict = cPickle.load(f)

                # Load only two classes, and code as binary one-hot labels.
                labels = np.array(dict['labels'])
                indices_0 = np.where(labels == label_0)[0] 
                indices_1 = np.where(labels == label_1)[0] 
                all_ind_0 = np.concatenate((all_ind_0, indices_0 + 10000 * i)).astype(int)
                all_ind_1 = np.concatenate((all_ind_1, indices_1 + 10000 * i)).astype(int)
                all_images[10000 * i : 10000 * (i + 1), :] = dict['data']
                print('done with batch {}, added {} to c0 and {} to c1'.format(
                    i, len(indices_0), len(indices_1)))

            images_0 = np.array([all_images[i] for i in all_ind_0])  # (data_size, 3072)
            images_1 = np.array([all_images[i] for i in all_ind_1])

        elif split == 'test':
            path = 'data/cifar-10-batches-py/'
            f = open(path + 'test_batch', 'rb')
            dict = cPickle.load(f)

            # Load only two classes, and code as binary one-hot labels.
            labels = np.array(dict['labels'])
            indices_0 = np.where(labels == label_0)[0] 
            indices_1 = np.where(labels == label_1)[0] 

            # Fetch images, resize to 28x28, and flatten again.
            images_0 = np.array([dict['data'][i] for i in indices_0])  # (data_size, 3072)
            images_1 = np.array([dict['data'][i] for i in indices_1])


    def resize28(img):
        img = flat_to_img(img, 32)
        img = imresize(img, (28, 28, 3))
        return img
    images_0 = map(resize28, images_0)  # (data_size, 28, 28, 3)
    images_1 = map(resize28, images_1)
    images_0 = np.array(images_0).transpose([0, 3, 1, 2])
    images_1 = np.array(images_1).transpose([0, 3, 1, 2])

    # Make one-hot labels.
    labels_0 = np.tile([1., 0.], [len(images_0), 1])  # (data_size, 2)
    labels_1 = np.tile([0., 1.], [len(images_1), 1]) 
    
    return images_0, images_1, labels_0, labels_1 

def flat_to_img(img_flat, img_d, fname=None, save=False):
    """Saves a pixel data blob as an image file.
    """
    assert len(img_flat.shape) == 1, 'image must be flat'

    img = np.reshape(img_flat, [3, img_d, img_d]).transpose([1, 2, 0])
    if save:
        imsave(fname, img)
    return img


def img_to_flat(img):
    if img.shape[2] == 3:  # HWC
        return np.reshape(img.transpose([2, 0, 1]), -1)  # Channel-first order.
    elif img.shape[0] == 3:  # CHW
        return np.reshape(img, -1)
    return flat 



def create_datasets(imagearray, labelarray):
    train_set_x = np.empty((200,3072))
    train_set_y = np.empty((1,200),dtype=np.int16)

    i = 0
    j = 0
    while (j < 200):                #   200 train images
        x = random.randint(0,1)
        if (labelarray[i] == 3):    #   Cats
            train_set_x[j] = imagearray[i]
            train_set_y[0,j] = 1    #   Cat is True
            j+=1
        elif (x % 2 == 0 and labelarray[i] != 3):   #    NOT Cats
            train_set_x[j] = imagearray[i]
            train_set_y[0,j] = 0    # Cat is False
            j+=1
        i+=1
        
    train_set_x = train_set_x.T     #   Reshape to (3072, 200) 
    
    test_set_x = np.empty((50,3072))                #   50 test images
    test_set_y = np.empty((1,50),dtype=np.int16)

    i = 0
    j = 0
    while (j < 50):
        x = random.randint(0,1)
        if (labelarray[9999-i] == 3):#  In Reverse Order is Cat
            test_set_x[j] = imagearray[9999-i]
            test_set_y[0,j] = 1
            j+=1
        elif (x % 2 == 0 and labelarray[i] != 3):
            test_set_x[j] = imagearray[9999-i]
            test_set_y[0,j] = 0
            j+=1
        i+=1

    test_set_x = test_set_x.T       #   Reshape to (3072, 50)

    train_set_x = train_set_x/255.  #   0-255 -> 0-1
    test_set_x = test_set_x/255.

    return train_set_x, train_set_y, test_set_x, test_set_y
