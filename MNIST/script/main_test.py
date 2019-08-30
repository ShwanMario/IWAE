
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
parser.add_argument("--batch_size",type=int,
                    default=1,
                    help="""batch size during the training process""")
parser.add_argument("--epoch",type=int,
                    required=True,
                    help="""which epoch weights to use""")
parser.add_argument("--p_x",type=str,
                    default="discrete",
                    help="""datatype of x""")
parser.add_argument("--device",type=str,
                    default="gpu",
                    help="""cpu or gpu""")

args = parser.parse_args()

num_samples=args.num_m*args.num_k
## read data
with open('../data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
    

test_image = data['test_image']
test_label = data['test_label']

batch_size = args.batch_size
test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

if args.num_stochastic_layers == 1:
    vae = IWAE_1(args.latent_variable_2, args.size_input,p_x=args.p_x)
elif args.num_stochastic_layers == 2:
    vae = IWAE_2(args.latent_variable_1, args.latent_variable_2, args.size_input,p_x=args.p_x)
    
vae.double()
if args.device=='gpu':
    vae.cuda()
model_file_name = ("../output/model/"
                   "{}_layers_{}_m_{}_k_{}_epoch_{}.model").format(
                       args.model, args.num_stochastic_layers,
                       args.num_m,args.num_k,args.epoch)
vae.load_state_dict(torch.load(model_file_name))

tot_loss = 0
tot_size = 0

for idx, data in enumerate(test_data_loader):
    print(idx)
    data = data.double()
    with torch.no_grad():
        inputs = Variable(data).to(args.device)
        if args.model == "IWAE":
            inputs = inputs.expand(num_samples, batch_size, args.size_input)
        elif args.model=='VAE':
            inputs = inputs.repeat(num_samples, 1)
            inputs = inputs.expand(1, batch_size * num_samples, args.size_input)  ################
        loss= vae.test_loss(inputs,args.num_m,args.num_k)

        size = inputs.size()[0]
        tot_size += size
        print(loss.item())
        tot_loss += loss.item() * size
        print("Average loss: {:.2f}".format(tot_loss / tot_size))

print(model_file_name, end = ",")        
print("Average loss: {:.2f}".format(tot_loss/tot_size))
