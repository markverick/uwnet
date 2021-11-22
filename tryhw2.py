from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(4),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .2
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

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# Without batch normalization, rate 0.01:
# ('training accuracy: %f', 0.41071999073028564)
# ('test accuracy:     %f', 0.41269999742507935)
# Without batch normalization, rate 0.05:
# ('training accuracy: %f', 0.4722599983215332)
# ('test accuracy:     %f', 0.461899995803833)

# With batch normalization, rate 0.003:
# ('training accuracy: %f', 0.47756001353263855)
# ('test accuracy:     %f', 0.47350001335144043)
# With batch normalization, rate 0.01:
# ('training accuracy: %f', 0.5364599823951721)
# ('test accuracy:     %f', 0.5322999954223633)
# With batch normalization, rate 0.05:
# ('training accuracy: %f', 0.5394600033760071)
# ('test accuracy:     %f', 0.529699981212616)
# With batch normalization, rate 0.2:
# ('training accuracy: %f', 0.32330000400543213)
# ('test accuracy:     %f', 0.3285999894142151)

# The accuracy slightly increases a with batch normalization

# Notice that the accuracy of the covnet with batch normalization at learning rate
# 0.01 and 0.05 is very similar, but those of without batch normalization
# differs by a large aount. Even LR=0.003 is not much different from LR=0.01

# It seems that without batch normalization, it suffers more from overfitting.
# So, it won't find any better model after a number of epoch.
# On the other hand, the one with batch normalization seems to converge much slower.
# and produce a better result.