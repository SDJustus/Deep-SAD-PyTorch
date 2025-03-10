import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class Custom_LeNet(BaseNet):

    def __init__(self, img_size, rep_dim=128):
        super().__init__()
        self.img_size = img_size
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        #self.conv6 = nn.Conv2d(512, 1024, 5, bias=False, padding=2)
        #self.bn2d6 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        #self.fc1 = nn.Linear(1024 * 4 * 4, self.rep_dim, bias=False)
        self.fc1 = nn.Linear(512 * 8 * 8, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 3, self.img_size, self.img_size)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        #x = self.conv6(x)
        #x = self.pool(F.leaky_relu(self.bn2d6(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class Custom_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (8 * 8)), 512, 5, bias=False, padding=2)
        #self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 1024, 5, bias=False, padding=2)
        #nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        #self.bn2d1 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        #self.deconv2 = nn.ConvTranspose2d(1024, 512, 5, bias=False, padding=2)
        #nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv7 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv7.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (8 * 8)), 8, 8)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        #x = F.interpolate(F.leaky_relu(self.bn2d1(x)), scale_factor=2)
        #x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d2(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv6(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv7(x)
        x = torch.sigmoid(x)

        return x


class Custom_LeNet_Autoencoder(BaseNet):

    def __init__(self,  img_size, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = Custom_LeNet(rep_dim=rep_dim, img_size=img_size)
        self.decoder = Custom_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
