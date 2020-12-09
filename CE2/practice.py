import cv2 as cv
import numpy as np
import pyautogui
import pygame
import matplotlib.pyplot as plt
from IPython.display import display, Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
########################################################################
########                       LEARNING                          #######
########################################################################
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

train_image = trainset.data.numpy()
train_label = trainset.targets.numpy()

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    ###################################################################################
    #                                   YOUR CODE HERE                                #
    ###################################################################################
    x_reshaped = np.reshape(x, (np.shape(x)[0],-1))
    out = np.matmul(x_reshaped,w) + b
    ###################################################################################
    #                                  END OF YOUR CODE                               #
    ###################################################################################
    cache = (x, w, b)
    return out, cache

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###################################################################################
    #                                   YOUR CODE HERE                                #
    ###################################################################################
    zeros = np.zeros(np.shape(x))
    out = np.maximum(zeros,x)
    ###################################################################################
    #                                  END OF YOUR CODE                               #
    ###################################################################################
    cache = x
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    ###################################################################################
    #                                   YOUR CODE HERE                                #
    ###################################################################################
    dx = np.reshape(np.matmul(dout,np.transpose(w)), np.shape(x))
    dw = np.matmul(np.transpose(np.reshape(x,(np.shape(x)[0],-1))),dout)
    db = np.sum(dout, axis = 0)
    ###################################################################################
    #                                  END OF YOUR CODE                               #
    ###################################################################################
    
    return dx, dw, db

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    ###################################################################################
    #                                   YOUR CODE HERE                                #
    ###################################################################################
    dx = np.array(x>0, dtype=float)*dout
    ###################################################################################
    #                                  END OF YOUR CODE                               #
    ###################################################################################
    return dx

####### Learning ########
class FC_user_model(nn.Module):
    def __init__(self):
        super(FC_user_model, self).__init__()
        fc1 = nn.Linear(28*28, 200)
        fc2 = nn.Linear(200, 100)
        fc3 = nn.Linear(100,10)
        
        self.fc_module = nn.Sequential(
            nn.Flatten(), fc1, nn.ReLU(), fc2, nn.ReLU(), fc3     
        )
        
    def forward(self, x):
        out = self.fc_module(x)
        return out
    
# Write your model name here
model = FC_user_model()

# You can change criterion, optimizer, num_epochs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10

# This is for training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    print('\nEpoch: %d' % (epoch+1))
    for batch_idx, data in enumerate(trainloader):
        image, label = data
        # Grad initialization
        optimizer.zero_grad()
        # Forward propagation
        output = model(image)
        # Calculate loss
        loss = criterion(output, label)
        # Backprop
        loss.backward()
        # Weight update
        optimizer.step()
        
        train_loss += loss.item()
        _,predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        if (batch_idx+1) % 1000 == 0:
            print("Step: {}/{} | train_loss: {:.4f} | Acc:{:.3f}%".format(batch_idx+1, len(trainloader), train_loss/1000, 100.*correct/total))

# Create Window.
cv.namedWindow("Number");
cv.moveWindow("Number", 2000, 1000);

########################################################################
Resolution_X = 2560
Resolution_Y = 1440

# Find Paint Window First.
icon_template = cv.imread('icon.png', cv.IMREAD_COLOR)
icon_template = np.array(icon_template)
pic = pyautogui.screenshot(region=(0, 0, Resolution_X , Resolution_Y))
img_frame = np.array(pic)
img_frame  = cv.cvtColor(img_frame, cv.COLOR_RGB2BGR)
meth = 'cv.TM_CCOEFF'
method = eval(meth)
res = cv.matchTemplate(icon_template, img_frame, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
paint_top_left = cv.minMaxLoc(res)[3]+np.array([4, 158])

# Finally, Zoom the skill icon.
while 1:
    
    pic = pyautogui.screenshot(region=(paint_top_left[0], paint_top_left[1], 500, 500))
    img_frame  = cv.cvtColor(np.array(pic), cv.COLOR_RGB2BGR)
    img_frame = cv.resize(img_frame, dsize=(28, 28), interpolation=cv.INTER_LINEAR)
    cv.imshow("Number", img_frame)
    data = np.zeros([1,1,28,28], dtype=float)
    data[0][0] = img_frame[:,:,2]
    data = torch.from_numpy(data).float()

    ######################
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        output = model(data)
        _,predicted = output.max(1)
        print(predicted)
    ######################
    key = cv.waitKey(1)
    if key == 27:
        break
