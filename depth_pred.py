import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os,time,cv2, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import subprocess
import utils
from nyu_dataloader import NYUDataset
from dense_to_sparse import UniformSampling

sys.path.append("models")
from RefineNet import build_refinenet

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])

num_samples = 100
traindir = "/media/mawe/40E20648E2064320/nyudepthv2"
model = "RefineNet-Res101"
dataset = "nyudepthv2"
continue_training = False
mode = "train"
num_epochs = 20
batch_size = 8

sparsifier = UniformSampling(num_samples=num_samples)
train_dataset = NYUDataset(traindir, type='train', modality="rgbd", sparsifier=sparsifier)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

if "Res50" in model and not os.path.isfile("models/resnet_v2_50.ckpt"):
    download_checkpoints("Res50")
if "Res101" in model and not os.path.isfile("models/resnet_v2_101.ckpt"):
    download_checkpoints("Res101")
if "Res152" in model and not os.path.isfile("models/resnet_v2_152.ckpt"):
    download_checkpoints("Res152")

# Compute your softmax cross entropy loss
print("Preparing the model ...")
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,1])

network, init_fn = build_refinenet(net_input, num_classes=1)

#define loss function

losses = tf.losses.absolute_difference(network, net_output)
loss = tf.reduce_mean(losses)

opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = "checkpoints/latest_model_" + model + "_" + dataset + ".ckpt"
if continue_training or not mode == "train":
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

avg_scores_per_epoch = []

avg_loss_per_epoch = []

for epoch in range(0, num_epochs):
    current_losses = []

    cnt = 0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_dataset))

    num_iters = int(np.floor(len(id_list) / batch_size))
    st = time.time()
    epoch_st = time.time()

    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(batch_size):
            index = i * batch_size + j
            id = id_list[index]
            input_np, depth_np = train_dataset.__getitem__(id)
            input_np = np.delete(input_np, 0, 2)

            #with tf.device('/cpu:0'):

            input_image_batch.append(np.expand_dims(input_np, axis=0))
            output_image_batch.append(np.expand_dims(depth_np, axis=0))

        # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
        # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0,
        #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
        # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0,
        #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
        # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****

        # memory()

        if batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
            output_image_batch = output_image_batch.reshape(output_image_batch.shape + (1,))

        # Do the training
        _, current = sess.run([opt, loss], feed_dict={net_input: input_image_batch, net_output: output_image_batch})
        current_losses.append(current)
        cnt = cnt + batch_size
        if cnt % 2*batch_size == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (
            epoch, cnt, current, time.time() - st)
            utils.LOG(string_print)
            st = time.time()