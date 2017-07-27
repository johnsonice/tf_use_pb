#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:30:31 2017

@author: chengyu
"""

import tensorflow as tf 
import json
import cv2
from postprocess import postprocess

#%%
## preprocess function 
def preprocess(im,meta):
    h, w, c = meta['inp_size']
    imsz = cv2.resize(im,(w,h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz

#%%
## read tf model from pb 
tf.reset_default_graph()

pb_file = 'yolo.pb'
pb_meta = 'yolo.meta'
img = 'sample_dog.jpg'

with tf.gfile.FastGFile(pb_file, "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
    
tf.import_graph_def(graph_def,name="")
with open(pb_meta, 'r') as fp:
	meta = json.load(fp)
    
#%%
## get input and out put node 
sess = tf.Session()
inp = tf.get_default_graph().get_tensor_by_name('input:0')
out = tf.get_default_graph().get_tensor_by_name('output:0')

#%%
## use model and input, output node for inference 
pic = cv2.imread(img,1)
x = preprocess(pic,meta)
feed_dict = {inp: [x]}
net_out = sess.run(out, feed_dict)


############################################################
# waiting to be finished
meta['thresh'] = 0.3
result,boxes = postprocess(net_out,meta,pic)

###########


#%%
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
#%%


## talke a look at node names ------------ this is not revelent 
graph = tf.get_default_graph()
node_names = [n.name for n in graph.as_graph_def().node]
