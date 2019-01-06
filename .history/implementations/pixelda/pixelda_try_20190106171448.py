import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from mnistm import MNISTM

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_residual_blocks', type=int, default=1, help='number of residual blocks in generator')
parser.add_argument('--latent_dim', type=int, default=10, help='dimensionality of the noise input')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')
parser.add_argument('--sample_interval', type=int, default=300, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2**4)
patch = (1, patch, patch)

cuda = True if torch.cuda.is_available() else False

print("cuda : {}".format(cuda))

def weights_init_normal(m):
    classname = m.__class__.__name__
    print("classname : {}".format(classname))
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock1(nn.Module):
    def __init__(self, in_features=32, out_features=64, kernel_size=3, stride=2, padding=1):
        super(ResidualBlock1, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size= kernel_size, stride = stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        encode_x = self.block(x)
        return x, encode_x


class ResidualBlock2(nn.Module):
    def __init__(self, in_features=64, out_features=128, kernel_size=3, stride=2, padding=1):
        super(ResidualBlock2, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size= kernel_size, stride = stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        encode_x = self.block(x)
        return x, encode_x


class ResidualBlock1(nn.Module):
    def __init__(self, in_features=32, out_features=64, kernel_size=3, stride=2, padding=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size= kernel_size, stride = stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        encode_x = self.block(x)
        return x, encode_x


class source_encode_ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(source_encode_ResidualBlock, self).__init__()
        
        ### ENCODER
        self.encode_block = nn.Sequential(
            nn.Conv2d(in_channels=1*in_features,out_channels=4*in_features,kernel_size=(3, 3),stride=(2, 2),padding=0),
            nn.BatchNorm2d(4*in_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=4*in_features,out_channels=8*in_features,kernel_size=(3, 3),stride=(2, 2),padding=1),
            nn.BatchNorm2d(8*in_features),
            nn.LeakyReLU(inplace=True)
        )
        
        
    def forward(self, x):
        encode_x = self.encode_block(x)
        return x, encode_x    


class target_encode_ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(target_encode_ResidualBlock, self).__init__()
        
        ### ENCODER
        self.encode_block = nn.Sequential(
            nn.Conv2d(in_channels=1*in_features,out_channels=4*in_features,kernel_size=(3, 3),stride=(2, 2),padding=0),
            nn.BatchNorm2d(4*in_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=4*in_features,out_channels=8*in_features,kernel_size=(3, 3),stride=(2, 2),padding=1),
            nn.BatchNorm2d(8*in_features),
            nn.LeakyReLU(inplace=True)
        )
        
        
    def forward(self, x):
        encode_x = self.encode_block(x)
        return x, encode_x    

class decode_ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(decode_ResidualBlock, self).__init__()

        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8*in_features,out_channels=4*in_features,kernel_size=(3, 3),stride=(2, 2), padding=0),
            nn.BatchNorm2d(4*in_features),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=4*in_features,out_channels=1*in_features,kernel_size=(3, 3),stride=(2, 2),padding=1),
            nn.BatchNorm2d(1*in_features),
            nn.LeakyReLU(inplace=True),
            
        )

    def forward(self, encode_x):
        decode_x = self.decode_block(encode_x)
        decode_x = decode_x[:, :, :-1, :-1]
        decode_x = F.sigmoid(decode_x)
        return decode_x   


class target_encode_Generator(nn.Module):
    def __init__(self):
        super(target_encode_Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels*opt.img_size**2)
        self.l1 = nn.Sequential(nn.Conv2d(opt.channels*2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(target_encode_ResidualBlock())
        self.encode_resblocks = nn.Sequential(*resblocks)


    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        x, encode_out = self.encode_resblocks(out)


        return x, encode_out


class source_encode_Generator(nn.Module):
    def __init__(self):
        super(source_encode_Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels*opt.img_size**2)
        self.l1 = nn.Sequential(nn.Conv2d(opt.channels*2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(source_encode_ResidualBlock())
        self.encode_resblocks = nn.Sequential(*resblocks)


    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        x, encode_out = self.encode_resblocks(out)


        return x, encode_out


class decode_Generator(nn.Module):
    def __init__(self):
        super(decode_Generator, self).__init__()

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(decode_ResidualBlock())
        self.decode_resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())


    def forward(self, img, encode_out):
        out = img + self.decode_resblocks(encode_out)
        img_ = self.l2(out)

        return img_


class encode_Discriminator(nn.Module):
    def __init__(self):
        super(encode_Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True) ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(512, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, encode_x):
        validity = self.model(encode_x)

        return validity

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True) ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True) ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512)
        )

        input_size = opt.img_size // 2**4
        self.output_layer = nn.Sequential(
            nn.Linear(512*input_size**2, opt.n_classes),
            nn.Softmax()
        )

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label

# Loss function
adversarial_loss = torch.nn.MSELoss()
encode_adversarial_loss = torch.nn.MSELoss()
task_loss = torch.nn.CrossEntropyLoss()

# Loss weights
lambda_adv =  1
lambda_task = 0.1

# Initialize generator and discriminator
target_encode_generator = target_encode_Generator()
source_encode_generator = source_encode_Generator()
decode_generator = decode_Generator()

encode_discriminator = encode_Discriminator()
discriminator = Discriminator()
classifier = Classifier()

if cuda:
    target_encode_generator.cuda()
    source_encode_generator.cuda()
    decode_generator.cuda()

    encode_discriminator.cuda()
    discriminator.cuda()
    classifier.cuda()
    adversarial_loss.cuda()
    encode_adversarial_loss.cuda()
    task_loss.cuda()

# Initialize weights
target_encode_generator.apply(weights_init_normal)
source_encode_generator.apply(weights_init_normal)
decode_generator.apply(weights_init_normal)
encode_discriminator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader_A = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

os.makedirs('../../data/mnistm', exist_ok=True)
dataloader_B = torch.utils.data.DataLoader(
    MNISTM('../../data/mnistm', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers

optimizer_G = torch.optim.Adam( itertools.chain(target_encode_generator.parameters(), 
                                source_encode_generator.parameters(),
                                decode_generator.parameters(),
                                classifier.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(itertools.chain(encode_discriminator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):

        batch_size = imgs_A.size(0)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)
        encode_valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        encode_fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs_A      = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
        labels_A    = Variable(labels_A.type(LongTensor))
        imgs_B      = Variable(imgs_B.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise
        z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        imgs_A_x, encode_fake_B = source_encode_generator(imgs_A, z)
        decode_fake_B = decode_generator(imgs_A_x, encode_fake_B)

        # Perform task on translated source image
        label_pred = classifier(decode_fake_B)

        # Calculate the task loss
        task_loss_ =    (task_loss(label_pred, labels_A) + \
                        task_loss(classifier(imgs_A), labels_A)) / 2
        
        # Loss measures generator's ability to fool the discriminator
        g_loss =    lambda_adv * adversarial_loss(discriminator(decode_fake_B), valid) + \
                    0.1 * encode_adversarial_loss(encode_discriminator(encode_fake_B), encode_valid) + \
                    lambda_task * task_loss_

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        imgs_B_x, encode_real_B = source_encode_generator(imgs_B, z)
        decode_real_B = decode_generator(imgs_B_x, encode_real_B)
        # Measure discriminator's ability to classify real from generated samples
        encode_real_loss = encode_adversarial_loss(encode_discriminator(encode_real_B), encode_valid)
        encode_fake_loss = encode_adversarial_loss(encode_discriminator(encode_fake_B.detach()), encode_fake)
        decode_real_loss = adversarial_loss(discriminator(decode_real_B), valid)
        decode_fake_loss = adversarial_loss(discriminator(decode_fake_B.detach()), fake)
        encode_d_loss = (encode_real_loss + encode_fake_loss) / 2
        decode_d_loss = (decode_real_loss + decode_fake_loss) / 2
        d_loss = encode_d_loss + decode_d_loss

        d_loss.backward()
        optimizer_D.step()

        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        acc = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.numpy())
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]" %
                                                            (epoch, opt.n_epochs,
                                                            i, len(dataloader_A),
                                                            d_loss.item(), g_loss.item(),
                                                            100*acc, 100*np.mean(task_performance),
                                                            100*target_acc, 100*np.mean(target_performance)))

        batches_done = len(dataloader_A) * epoch + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((imgs_A.data[:5], decode_fake_B.data[:5], imgs_B.data[:5]), -2)
            save_image(sample, 'images/%d.png' % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)
