import os
# import pdb
from PIL import Image
from glob import glob
import tensorflow as tf
import pdb

def get_loader(root, batch_size, source_mix, classes, split_name = None, data_format='NHWC', seed=None):
    # This is only for MNIST
    n_classes = len(classes)
    channels = 1
    scale_size = 28
    if split_name:
        root = os.path.join(root, 'splits', split_name)
    frozen_classes = frozenset(classes)
    paths = [root + '/' + path for path in os.listdir(root) if path[-4:] == '.jpg' and int(path[0]) in frozen_classes]
    tf_decode = tf.image.decode_jpeg

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, channels]

    indices = [[] for j in range(n_classes)]
    class_set = frozenset(classes)  # for speed
    class_counts = [0] * n_classes
    class_to_index = {classes[j]:j for j in range(n_classes)}
    for idx in xrange(len(paths)):
        fn = paths[idx].split('/')[-1]
        f_class = int(fn[0])
        if f_class in class_set:
            class_counts[class_to_index[f_class]] += 1
            indices[class_to_index[f_class]].append((idx, f_class))

    source_n = int(min([class_counts[j] / source_mix[j] for j in range(n_classes)]))
    source_n_by_class = [int(source_mix[j] * source_n) for j in range(n_classes)]
    new_paths = [paths[idx[0]] for j in range(n_classes) for idx in indices[j][:source_n_by_class[j]]]
    class_assignments = [classes[j] for j in range(n_classes) for c in xrange(source_n_by_class[j])]

    # Print some diagnostics
    print('Paths contain', [len([1 for path in new_paths if int(path.split('/')[-1][0]) == j]) for j in classes], 'of', classes, 'respectively')
    print('Class assignments contain', [len([c for c in class_assignments if c == j]) for j in classes], 'of', classes, 'respectively')
    print('First 10 path/class pairs:', zip([path.split('/')[-1][0] for path in new_paths[:10]], class_assignments[:10]))

    filename_queue = tf.train.string_input_producer(list(new_paths), shuffle=True, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=channels)
    # label = tf.string_to_number(tf.substr(filename, 0, 1), out_type=tf.int32)
    image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = batch_size * 2
    capacity = min_after_dequeue + 3 * batch_size

    # images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=4, capacity=capacity, 
    #                               min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    images = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=capacity, 
                                   min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    images = tf.image.resize_nearest_neighbor(images, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(images, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    # base_onehot = tf.gather(tf.eye(n_classes, dtype=images.dtype), indices=classes, axis=1)
    # labels_onehot = tf.gather(tf.eye(n_classes), indices=labels, axis=0)

    return tf.to_float(images)  #, labels
