
import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../../model/")
from vae_models import *
from sys import exit
import argparse

parser = argparse.ArgumentParser(description="Importance Weighted Auto-Encoder")
parser.add_argument("--model", type = str,
                    choices = ["IWAE", "VAE"],
                    required = True,
                    help = "choose VAE or IWAE to use")
parser.add_argument("--num_stochastic_layers", type = int,
                    choices = [1, 2],
                    required = True,
                    help = "num of stochastic layers used in the model")
parser.add_argument("--num_m", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of ELBO""")
parser.add_argument("--num_k",type=int,
                    required=True,
                    help="""num of samples used in importance weighted ELBO""")
parser.add_argument("--size_input",type=int,
                    default=784,
                    help="""size of input data""")
parser.add_argument("--latent_variable_2",type=int,
                    default=50,
                    help="""number of neurons in latent variable h2""")
parser.add_argument("--latent_variable_1",type=int,
                    default=100,
                    help="""number of neurons in latent variable h1""")
parser.add_argument("--hidden_layer_1",type=int,
                    default=200,
                    help="""number of neurons in hidden layer 1""")
parser.add_argument("--hidden_layer_2",type=int,
                    default=100,
                    help="""number of neurons in hidden layer 2""")
parser.add_argument("--num_epoches",type=int,
                    default=5000,
                    help="""number of epoches""")
parser.add_argument("--batch_size",type=int,
                    default=1000,
                    help="""batch size during the training process""")
parser.add_argument("--p_x",type=str,
                    default="discrete",
                    help="""datatype of x""")
parser.add_argument("--device",type=str,
                    default="gpu",
                    help="""cpu or gpu""")
parser.add_argument("--learning_rate",type=float,
                    default=0.0001,
                    help="""learning rate""")
args = parser.parse_args()

num_samples=args.num_m*args.num_k
## read data
with open('../data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = args.batch_size
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

if args.num_stochastic_layers == 1:
    vae = IWAE_1(args.latent_variable_2, args.size_input,p_x=args.p_x,hidden_layer=args.hidden_layer_1)
elif args.num_stochastic_layers == 2:
    vae = IWAE_2(args.latent_variable_1, args.latent_variable_2, args.size_input,p_x=args.p_x,hidden_layer_1=args.hidden_layer_1,hidden_layer_2=args.hidden_layer_2)
    
vae.double() #cast all the floating point parameters and buffers to double datatype
if args.device=='gpu':
    vae.cuda()

optimizer = optim.Adam(vae.parameters(),lr=args.learning_rate)
num_epoches = args.num_epoches
train_loss_epoch = []
for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        data = data.double()
        inputs = Variable(data).to(args.device)
        if args.model == "IWAE":
            inputs = inputs.expand(num_samples, batch_size, args.size_input)##################
        elif args.model == "VAE":
            inputs = inputs.repeat(num_samples, 1)
            inputs = inputs.expand(1, batch_size*num_samples, args.size_input)################
        optimizer.zero_grad()
        loss= vae.train_loss(inputs,args.num_m,args.num_k)
        loss.backward()
        optimizer.step()
        print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}")
              .format(epoch, idx, loss.item()), flush = True)
        running_loss.append(loss.item())


    train_loss_epoch.append(np.mean(running_loss))

    if (epoch + 1) % 1000 == 0:
        torch.save(vae.state_dict(),
                   ("../output/model/{}_layers_{}_m_{}_k_{}_epoch_{}.model")
                   .format(args.model, args.num_stochastic_layers,
                           args.num_m,args.num_k, epoch))

