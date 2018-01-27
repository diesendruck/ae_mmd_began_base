import numpy as np
import os
import pdb
from PIL import Image
from glob import glob
import tensorflow as tf


def get_loader(root, batch_size, scale_size, data_format, split_name=None,
        is_grayscale=False, seed=None, target=None, n=None, mix=None):


    def load_labels(label_file, label_choices=None):
        lines = open(label_file, 'r').readlines()
        label_names = lines[1].strip().split(' ')
        if label_choices is None:
            label_choices = range(len(label_names))
        n_labels = len(label_choices)
        label_names = [label_names[choice] for choice in label_choices]
        print('Labels:')
        for ln in xrange(n_labels):
            print '  {:2d} {:2d} {}'.format(ln, label_choices[ln], label_names[ln])
        
        file_to_idx = {lines[i][:10]: i for i in xrange(2, len(lines))}
        labels = np.array([[int(it) for it in line.strip().split(
            ' ')[1:] if it != ''] for line in lines[2:]])[:, label_choices]
        return labels, label_names, file_to_idx


    def img_and_lbl_queue_setup(filenames, labels):
        data_dir = os.path.dirname(filenames[0])
        labels_tensor = tf.constant(labels, dtype=tf.float32)
        
        filenames_tensor = tf.constant(filenames)
        fnq = tf.RandomShuffleQueue(capacity=200, min_after_dequeue=100, dtypes=tf.string)
        fnq_enq_op = fnq.enqueue_many(filenames_tensor)
        filename = fnq.dequeue()
        
        reader = tf.WholeFileReader()
        flnm, data = reader.read(fnq)
        image = tf_decode(data, channels=3)
        image.set_shape([178, 218, 3])
        image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)
        image = tf.to_float(image)
        
        image_index = [tf.cast(tf.string_to_number(tf.substr(flnm, len(data_dir), 6)) - 1, tf.int32)]
        image_labels = tf.reshape(tf.gather(labels_tensor, indices=image_index, axis=0), [2])
        imq = tf.RandomShuffleQueue(capacity=60, min_after_dequeue=30, dtypes=[tf.float32, tf.float32],
                                    shapes=[[128, 128, 3], [2]])
        imq_enq_op = imq.enqueue([image, image_labels])
        imgs, img_lbls = imq.dequeue_many(batch_size)
        imgs = tf.image.resize_nearest_neighbor(imgs, size=[scale_size, scale_size])
        imgs = tf.subtract(1., tf.divide(imgs, 255.5))
        qr_f = tf.train.QueueRunner(fnq, [fnq_enq_op] * 3)
        qr_i = tf.train.QueueRunner(imq, [imq_enq_op] * 3)
        
        if data_format == 'NCHW':
            imgs = tf.transpose(imgs, [0, 3, 1, 2])
        return imgs, img_lbls, qr_f, qr_i


    dataset_name = os.path.basename(root)

    # Set channel size.
    if dataset_name == 'mnist':
        is_grayscale=True
        channels = 1
        scale_size = 28  # TODO: Determine whether scale should be 28.
    else:
        channels = 3
    
    # Define data path.
    if dataset_name in ['celeba', 'mnist'] and split_name:
        if target:
            data_path = os.path.join(root, 'splits', target)
        else:
            data_path = os.path.join(root, 'splits', split_name)

    # Collect files in target or split.
    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(data_path, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        
        if len(paths) != 0:
            break

    # Get image shape.
    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, channels]

    # Build queue for tuples of (images, labels).
    if dataset_name == 'celeba':
        # Get global labels.
        label_file = os.path.join(root, 'list_attr_celeba.txt')
        labels, label_names, file_to_idx = load_labels(label_file)  
        feature_id = label_names.index('Eyeglasses')
        labels = np.array(
            [[1.0, 0.0] if l[feature_id] == 1 else [0.0, 1.0] for l in labels])  # Binary one-hot labels.

        # For files in this target, get global indices.
        paths_and_global_indices = zip(paths, [file_to_idx[p[-10:]] for p in paths])

        # Fetch labels for paths in this split, using global indices.
        if target == 'test':
            # Make queue on all paths in 'test' target.
            imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(list(paths), labels)
            return imgs, img_lbls, qr_f, qr_i

        elif target == 'user':
            # Make queue on n paths, 50-50 from the two classes. 
            # First index corresponds to underrepresented class 0.
            paths_0 = ([p for p, gi in paths_and_global_indices if labels[gi][0] == 1])  # label [1.0, 0.0]
            paths_1 = ([p for p, gi in paths_and_global_indices if labels[gi][0] != 1])
            assert n / 2 < len(paths_0) and n / 2 < len(paths_1), 'asking for more than we have'
            paths = np.concatenate((
                np.random.choice(paths_0, n / 2),
                np.random.choice(paths_1, n / 2)))
            imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(list(paths), labels)
            return imgs, img_lbls, qr_f, qr_i

        elif target == 'train':
            # Make queue on n paths, 50-50 from the two classes. 
            paths_0 = ([p for p, gi in paths_and_global_indices if labels[gi][0] == 1])  # label [1.0, 0.0]
            paths_1 = ([p for p, gi in paths_and_global_indices if labels[gi][0] != 1])
            pcts = [int(mix[:2]), int(mix[2:])]
            larger_class_multiplier = float(max(pcts) / min(pcts))
            num_paths_1 = int(larger_class_multiplier * len(paths_0))
            assert num_paths_1 < len(paths_1), 'asking for more of class 1 than we have'
            paths = np.concatenate((paths_0, np.random.choice(paths_1, num_paths_1)))
            imgs, img_lbls, qr_f, qr_i = img_and_lbl_queue_setup(list(paths), labels)
            return imgs, img_lbls, qr_f, qr_i

    # Build queue for images.
    elif dataset_name == 'mnist':
        filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
        reader = tf.WholeFileReader()
        filename, data = reader.read(filename_queue)
        image = tf_decode(data, channels=channels)

        if is_grayscale:
            image = tf.image.rgb_to_grayscale(image)
        image.set_shape(shape)

        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * batch_size

        queue = tf.train.shuffle_batch(
            [image], batch_size=batch_size,
            num_threads=4, capacity=capacity,
            min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

        if dataset_name in ['celeba']:
            queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
            queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
        else:
            queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

        if data_format == 'NCHW':
            queue = tf.transpose(queue, [0, 3, 1, 2])
        elif data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(data_format))

        return tf.to_float(queue)


