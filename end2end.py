from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
import pdb
import time
import zmq
import dlib
from matplotlib import pyplot as plt
import align.detect_face

import argparse

GRAPH_PATH = '/home/wenfahu/faces/graph.pb'
detector = dlib.get_frontal_face_detector()

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def preprocess_image(imgs, image_size, _prewhiten=True):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    margin = 45

    nrof_samples = len(imgs)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in xrange(nrof_samples):
        img = imgs[i][..., ::-1]
        img_size = np.asarray(img.shape)[0:2]
        bboxes = detector(img, 1)
        bbox = bboxes[0]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det = det[index,:]
            det = np.squeeze(det)
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        # pdb.set_trace()
        # plt.imshow(aligned, interpolation='nearest')
        # plt.show()
            if _prewhiten:
                img = prewhiten(aligned)
            images[i,:,:,:] = img
        else:
            return None
    return images



with tf.gfile.FastGFile(GRAPH_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.InteractiveSession()

pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print('sever listening')
verification_threshold = 0.4
while True:
    msg = socket.recv()
    len1, row1, col1, len2, row2, col2 = np.fromstring(msg[:24], dtype=np.int32)
    print(np.fromstring(msg[:24], dtype=np.int32))
    img1 = np.fromstring(msg[24: 24 + len1], dtype=np.uint8)
    print (img1.shape)
    img1 = img1.reshape(row1, col1, -1)
    img2 = np.fromstring(msg[24 + len1:], dtype=np.uint8)
    print (img2.shape)
    img2 = img2.reshape(row2, col2, -1)

    imgs = preprocess_image([img1, img2], 160)
    if imgs is not None:
        feed_dict = {images_placeholder:imgs,  phase_train_placeholder:False}
        start = time.time()
        embd_feature = sess.run(embeddings, feed_dict=feed_dict)
        print(time.time() - start)
        features0 = embd_feature[0::2]
        features1 = embd_feature[1::2]
        diff = np.subtract(features0, features1)
        dist = np.sum(np.square(diff),1)
        print(dist)
        res = {}
        if dist < verification_threshold :
            res = 'acc'
        else:
            res = 'rej'
        socket.send(res)

'''
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    args = parser.parse_args()

    img_paths = [args.img_path]
    imgs = load_image(img_paths, 160)
    feed_dict = {images_placeholder:imgs,  phase_train_placeholder:False}
    start = time.time()
    embd_feature = sess.run(embeddings, feed_dict=feed_dict)
    print(time.time() - start)
    print (embd_feature)
    '''



# img_paths = ['/home/wenfahu/facenet/data/images/Anthony_Hopkins_0001.jpg',
# '/home/wenfahu/facenet/data/images/Anthony_Hopkins_0002.jpg',
# '/home/wenfahu/faces/facenet/data/images/Woody_Allen_0001.png',
# '/home/wenfahu/faces/facenet/data/images/Wally_Szczerbiak_0001.png']
