#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:30:31 2017

@author: chengyu
"""

import tensorflow as tf 
import json
import cv2
#from postprocess import postprocess
import numpy as np
from box import BoundBox, box_iou, prob_compare
import matplotlib.pyplot as plt
import math 
%matplotlib inline
#%%
## preprocess function 
def preprocess(im,meta):
    h, w, c = meta['inp_size']
    imsz = cv2.resize(im,(w,h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz

def expit(x):
    return 1./(1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def finxboxes(net_out,meta,h,w):
    H,W,_ = meta['out_size']
    threshold = meta['thresh']
    C,B = meta['classes'], meta['num']  ## number of classes(80 for yolo), number of boxes (5 for yolov2)
    anchors = meta['anchors']
    labels = meta['labels']
    net_out = net_out.reshape([H,W,B,-1])   ## reshape to 19,19,5,85 , 19X19 grade with 5 boxes for each grid, 
                                                    ## 85 premeters for each box
    boxes = list()
    for row in range(H):  # H = 19
        for col in range(W):  #W = 19
            for b in range(B):  # B = 5
                bx = BoundBox(C)  # a box instance with C=80 classes 
                bx.x,bx.y,bx.w,bx.h,bx.c = net_out[row,col,b,:5]  # assign first 5 as x,y,w,h,c
                bx.c = expit(bx.c)  # use logit, bounded between 0 and 1 
                bx.x = (col + expit(bx.x)) / W   # get the relative x position, between(0,1) 
                bx.y = (row + expit(bx.y)) / H   # get the relative y position, between(0,1) 
                bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W    #?
                bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H    #?
                classes = net_out_reshape[row,col,b,5:]
                bx.probs = _softmax(classes)*bx.c
                bx.probs *= bx.probs>threshold  # prob = prob if > threshod, otherwise 0 
                if sum(bx.probs)>0:             # i added this part, make it faster for inference , not good for training 
                    boxes.append(bx)
    
    # non max suppress boxes, delete overlapping boxes 
    for c in range(C):
        for i in range(len(boxes)):
            boxes[i].class_num = c 
        boxes = sorted(boxes,key=prob_compare)  # sort by higest class probability 
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.probs[c] == 0. :continue
            for j in range(i+1,len(boxes)):
                boxj = boxes[j]
                if box_iou(boxi,boxj)>=0.4:
                    boxes[j].probs[c] = 0.
    
    boxes = [box for box in boxes if sum(box.probs) > 0.]  ## i added this part, make it faster for inference , not good for training, get ride of all 0 probs boxes 
    if len(boxes)==0:
        return None
    
    boxes_out = list()
    for b in boxes:
        max_index = np.argmax(b.probs)
        max_prob = b.probs[max_index]
        label = 'object' * int(C < 2) 
        label += labels[max_index] * int(C>1) 
        left = int((b.x-b.w/2.)*w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        bo = {'label':label,'confidence':b.c,"topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}}
        boxes_out.append(bo)

    return boxes_out

def draw(pic,meta,boxes):
    pic = np.copy(pic)
    colors = meta['colors']
    labels = meta['labels']
    h,w,_ = pic.shape
    thick = int((h+w)/300)
    if boxes is None: return pic
    for b in boxes:
        x1 = b['topleft']['x']
        y1 = b['topleft']['y']
        x2 = b['bottomright']['x']
        y2 = b['bottomright']['y']
        label = b['label']
        max_index = labels.index(label)
        cv2.rectangle(pic,(x1,y1),(x2,y2),colors[max_index],thick)
        mess = '{}'.format(label)
        cv2.putText(pic,mess,(x1,y1-12),0, 1e-3 * h, colors[max_index],thick//3)

    return pic 

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
pic = cv2.imread(img)
x = preprocess(pic,meta)
feed_dict = {inp: [x]}
net_out = sess.run(out, feed_dict)

meta['thresh'] = 0.3  ## set threshold 
h,w,_ = pic.shape     ## get original pic shape 
boxes = finxboxes(net_out,meta,h,w)    # return list of box dict 
print(boxes)
# draw image
return_img = draw(pic,meta,boxes)
plt.imshow(return_img)

#%%
## spped test 
import timeit

def predict():
    pic = cv2.imread(img)
    x = preprocess(pic,meta)
    feed_dict = {inp: [x]}
    net_out = sess.run(out, feed_dict)
    
    meta['thresh'] = 0.3  ## set threshold 
    h,w,_ = pic.shape     ## get original pic shape 
    boxes = postprocess_json(net_out,meta,h,w) 
    return boxes

timeit.timeit(predict,number=100)  ## run 100 pictures 



# result = postprocess(net_out,meta,pic)
# plt.imshow(result)  ## BGR
## talke a look at node names ------------ this is not revelent 
# graph = tf.get_default_graph()
# node_names = [n.name for n in graph.as_graph_def().node]
