from uwnet import *

def conv_net():
    l = [   make_connected_layer(3072, 40),
            make_activation_layer(RELU),
            make_connected_layer(40, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer: 

# For default conv network
# training accuracy : 0.6817200183868408
# test accuracy     : 0.6394000053405762
#
# The total number of operations of the original covnet is: 
# CONV LAYER: 221184
# CONV LAYER: 294912
# CONV LAYER: 294912
# CONV LAYER: 294912
# CONNECTED LAYER: 327680
# Total counts: 1,105,920 + 327,680 = 1,433,600 Operations


# For the fully connected network
# training accuracy : 0.4361000061035156
# test accuracy     : 0.4146000146865845
# It is extremely hard for the fully connected network to match the performance of the convolutional
# network. With as small as 1.4 million operations, I can only put 2 layers of (3072, 40) and
# (40, 10). It needs a slightly more operations and has lower performance than 
# the convolutional network.

# The total number of maxtrix operations:
# CONNECTED LAYER: 15728640
# CONNECTED LAYER: 51200
# Total counts: 15728640 + 51200 = 15,779,840