import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import json

import data_loader as custom_dl
from layers import *
from modules import *

#root = '/ProxylessGAN'
root = './'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu=2
#device = torch.device("cpu")
print("using " + str(device))

print("n_gpu: ", ngpu)

first_conv = ConvLayer(3, 24, kernel_size=3, stride=2, dilation=1, groups=1, bias=False, has_shuffle=False, act_func="relu6")
feature_mix_layer = ConvLayer(20, 100, kernel_size=1, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False, act_func="relu6")
classifier = LinearLayer(100, 10)

def add_blocks(layer, in_channels, out_channels, stride):
    for kernel_size in [3, 5, 7]:
        for expand_ratio in [1, 3, 6]:
            blocks[layer].append(MobileInvertedResidualBlock(MBInvertedConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, expand_ratio=expand_ratio), None))
            if in_channels == out_channels:
                blocks[layer].append(MobileInvertedResidualBlock(MBInvertedConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, expand_ratio=expand_ratio),
                    IdentityLayer(in_channels, out_channels)))
#RIGHT NOW IT ONLY DOES RESIDUAL CONNECTIONS
# Plan to work on pooling and other connections tomorrow :)


n_blocks = 4
blocks = [[] for i in range(n_blocks)]

add_blocks(0, 24, 40, 1)
add_blocks(1, 40, 40, 1)
add_blocks(2, 40, 40, 1)
add_blocks(3, 40, 20, 2)

# Initialize block_probs to be an even distribution(1/n)
block_probs = [[] for i in range(n_blocks)]
for i in range(n_blocks):
    block_probs[i] = [(1/len(blocks[i])) for j in range(len(blocks[i]))]

# Initialize block_weights to all start at 0(used to be 1/n)
block_weights = [[] for i in range(n_blocks)]
for i in range(n_blocks):
    block_weights[i] = [0 for j in range(len(blocks[i]))]

train_loader, valid_loader = custom_dl.get_train_valid_loader(data_dir=root+'/data/cifar10/',
                                                              batch_size=16,
                                                              augment=False,
                                                              random_seed=1)
test_loader = custom_dl.get_test_loader(data_dir=root+'/data/cifar10/', batch_size=16)

def save_grad(module, gradInput, gradOutput):
    module.block_grad = gradInput[0]
    
def save_forward(module, forwardInput, output):
    module.block_forward = output[0]

def train_layer(data, i, blocks):
    net_blocks = []
    # For each block layer, chose a random block to train
    for block_layer in blocks:
        block_i = np.random.choice(len(block_layer))
        block = block_layer[block_i]
        net_blocks.append(block)
    net = ProxylessNasNet(first_conv, net_blocks, feature_mix_layer, classifier, ngpu=ngpu).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.weight_parameters(), lr=0.001)

    # get the inputs
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the gradient
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def train_architecture_params(data):
    # Choose a random layer
    block_layer_i = np.random.choice(len(blocks))
    # choose a random path for this layer
    block_a_index = np.random.choice(len(blocks[block_layer_i]), p=block_probs[block_layer_i])
    # choose another random path, ensuring that it does not match block_a_index
    block_b_index = block_a_index
    while block_a_index == block_b_index:
        block_b_index = np.random.choice(len(block_probs[block_layer_i]), p=block_probs[block_layer_i])
    
    block_a = blocks[block_layer_i][block_a_index]
    block_b = blocks[block_layer_i][block_b_index]

    blocks_a = []
    blocks_b = []
    for layer_i in range(len(blocks)):
        if layer_i == block_layer_i:
            blocks_a.append(block_a)
            blocks_b.append(block_b)
        else:
            rand_block_i = np.random.choice(len(blocks[layer_i]))
            blocks_a.append(blocks[layer_i][rand_block_i])
            blocks_b.append(blocks[layer_i][rand_block_i])

    # Register hooks for computing dLdg
    block_a_back_hook = block_a.register_backward_hook(save_grad)
    block_a_forward_hook = block_a.register_forward_hook(save_forward)
    block_b_back_hook = block_b.register_backward_hook(save_grad)
    block_b_forward_hook = block_b.register_forward_hook(save_forward)

    net_a = ProxylessNasNet(first_conv, blocks_a, feature_mix_layer, classifier, ngpu=ngpu).to(device)
    net_b = ProxylessNasNet(first_conv, blocks_b, feature_mix_layer, classifier, ngpu=ngpu).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # get the inputs
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward + backward net_a
    outputs = net_a(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

    # forward + backward net_b
    outputs = net_b(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

    dLdg_a = torch.sum(block_a.block_grad * block_a.block_forward).item()
    dLdg_b = torch.sum(block_b.block_grad * block_b.block_forward).item()

    # Compute the gradients of each block
    block_grad_a = 0
    block_grad_a += dLdg_a * (block_probs[block_layer_i][block_a_index] * (1. - block_probs[block_layer_i][block_a_index]))
    block_grad_a += dLdg_b * (block_probs[block_layer_i][block_a_index] * (0. - block_probs[block_layer_i][block_b_index]))
    
    block_grad_b = 0
    block_grad_b += dLdg_b * (block_probs[block_layer_i][block_b_index] * (1. - block_probs[block_layer_i][block_b_index]))
    block_grad_b += dLdg_a * (block_probs[block_layer_i][block_b_index] * (0. - block_probs[block_layer_i][block_a_index]))

    # Update the architecture params using simple SGD(but ascent in this case, because higher weight is better)
    lr = 1 # keep the learning rate large for now. Lower when I move to Adam(lr 0.006 in paper)
    block_weights[block_layer_i][block_a_index] += block_grad_a * lr
    block_weights[block_layer_i][block_b_index] += block_grad_b * lr

    # Free up space on GPU
    block_a.block_grad = None
    block_b.block_grad = None
    block_a.block_forward = None
    block_b.block_forward = None

    # Cleanup hooks
    block_a_forward_hook.remove()
    block_b_forward_hook.remove()
    block_a_back_hook.remove()
    block_b_back_hook.remove()
    return

block_loss = [0 for i in range(len(blocks))]
for epoch in range(50):
    for i, data in enumerate(train_loader, 0):
        
        # Train a block on the train data
        loss = train_layer(data, i, blocks)

        # Train the architecture params on heldout validation data
        train_architecture_params(next(iter(valid_loader)))

        block_loss[i % len(blocks)] = loss

        for block_layer in blocks:
            for block in block_layer:
                assert (block.block_grad is None and block.block_forward is None), "[!] WEE WOO WEE WOO"

        if i % 100 == 0:
            print("~[e%d]batch %d~"%(epoch, i))
            best_blocks = []
            for block_layer_i in range(len(blocks)):
                best_block_i = None
                for block_i in range(len(blocks[block_layer_i])):
                    if best_block_i is None or block_weights[block_layer_i][block_i] > block_weights[block_layer_i][best_block_i]:
                        best_block_i = block_i
                best_blocks.append(blocks[block_layer_i][best_block_i])
                print('[layer %d]\'s best block is [block %d]' %(block_layer_i, best_block_i))
            net = ProxylessNasNet(first_conv, best_blocks, feature_mix_layer, classifier, ngpu=ngpu).to(device)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            criterion = nn.CrossEntropyLoss()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            print('Best Net loss: %.3f'%loss.item())

print('Finished Training')

best_blocks = []
for block_layer_i in range(len(blocks)):
    best_block_i = None
    for block_i in range(len(blocks[block_layer_i])):
        if best_block_i is None or block_weights[block_layer_i][block_i] > block_weights[block_layer_i][best_block_i]:
            best_block_i = block_i
    best_blocks.append(blocks[block_layer_i][best_block_i])
    print('[layer %d]\'s best block is [block %d]' %(block_layer_i, best_block_i))
net = ProxylessNasNet(first_conv, best_blocks, feature_mix_layer, classifier, ngpu=ngpu).to(device)
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy w/ best blocks on test: %d %%' % (
    100 * correct / total))

def save_model(proxyless_model, name):
    config = proxyless_model.config
    with open(root + '/saved/proxyless_nas_' + name + '.config', 'w') as outfile:  
        json.dump(config, outfile)
    torch.save(proxyless_model.state_dict(), root + '/saved/proxyless_nas_' + name + '.pth')

save_model(net, 'cifar-10')

    
