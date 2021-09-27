import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
from PIL import Image
import natsort
import time

#======== 800x448 resnet ae ======================

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = models.resnet18(pretrained=False)
        self.num_feature = self.ResNet18.fc.in_features
        self.ResNet18.fc = nn.Linear(self.num_feature, 2 * self.z_dim)
        self.ResNet18.avgpool

    def forward(self, x):
        x = self.ResNet18(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class ResNet34Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet34Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet34 = models.resnet34(pretrained=False)
        self.num_feature = self.ResNet34.fc.in_features
        self.ResNet34.fc = nn.Linear(self.num_feature, self.z_dim)

    def forward(self, x):
        x = self.ResNet34(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return x

class ResNet50Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet50Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet50 = models.resnet50(pretrained=False)
        self.num_feature = self.ResNet50.fc.in_features
        self.ResNet50.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet50(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

class ResNet101Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet101Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet101 = models.resnet101(pretrained=False)
        self.num_feature = self.ResNet101.fc.in_features
        self.ResNet101.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet101(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, size=(14, 25), mode='bilinear', align_corners=False)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(224, 400), mode='bilinear', align_corners=False)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 448, 800)
        return x

class ResNet34Dec(nn.Module):

    def __init__(self, num_Blocks=[3, 4, 6, 3], z_dim=32, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, size=(14, 25), mode='bilinear', align_corners=False)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(224, 400), mode='bilinear', align_corners=False)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 448, 800)
        return x

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        """
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)
        
        self.encoder = ResNet34Enc(z_dim=z_dim)      
        self.encoder = ResNet50Enc(z_dim=z_dim)       
        self.encoder = ResNet101Enc(z_dim=z_dim)
        """
        self.encoder = ResNet34Enc(z_dim=z_dim)
        self.decoder = ResNet34Dec(z_dim=z_dim)

    def forward(self, x):
        z = self.encoder(x)
        # z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean

SparsityLoss = nn.L1Loss(reduction='sum')
def loss_func(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    L1loss = SparsityLoss(recon_x, x)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L1loss


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))


save_PATH = './Result/ADL-Rundle-6_resnet34_800x448_ae'
if not os.path.exists(save_PATH):
    os.makedirs(save_PATH)

epoch_num = 20
batch_size = 8
latent_dim = 32
learnR = 1e-3

exp_path = save_PATH + '/epoch{}_batch{}_z{}_lr{}'.format(epoch_num, batch_size, latent_dim, learnR)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

vae = VAE(z_dim=latent_dim).cuda()
optimizer = optim.Adam(vae.parameters(), lr=learnR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


transform = transforms.Compose([transforms.Resize([448, 800]),
                                transforms.ToTensor()])
# /home/fatih/mnt/fairmot_data/data/MOT17/images/test/MOT17-01-DPM/img1'
myDatasetTrain = CustomDataSet('/home/fatih/mnt/fairmot_data/data/MOT15/images/train/ADL-Rundle-6/img1', transform=transform)
myDatasetTest = CustomDataSet('/home/fatih/mnt/fairmot_data/data/MOT17/images/test/MOT17-01-DPM/img1', transform=transform)

# Data loader (input pipeline)
train_iter = DataLoader(myDatasetTrain, batch_size=batch_size, shuffle=True, drop_last=True)
test_iter = DataLoader(myDatasetTest, batch_size=batch_size, shuffle=False, drop_last=True)

"""
print('Training starts')
for epoch in range(0, epoch_num):
    time_start = time.time()
    l_sum = 0
    for x in train_iter:
        # x = torch.sigmoid(x).cuda()
        x = x.cuda()
        # print(x.requires_grad)
        optimizer.zero_grad()
        recon_x = vae.forward(x)
        loss = loss_func(recon_x, x)
        l_sum += loss
        loss.backward()
        optimizer.step()
    # scheduler.step()
    time_end = time.time()
    print('elapsed time (min) : %0.1f' % ((time_end - time_start) / 60))
    print('====> Epoch: %d elbo_Loss : %0.8f' % ((epoch + 1), l_sum / myDatasetTrain.__len__()))

torch.save(vae.state_dict(), exp_path + '/last_model.pth')
"""
vae.load_state_dict(torch.load(exp_path + '/last_model.pth'))

inx = 1
with torch.no_grad():
    for t_img in test_iter:
        t_img = Variable(t_img).cuda()
        result = vae.forward(t_img)
        for batches in range(batch_size):
            infer_out = result.data[batches]
            utils.save_image(infer_out, exp_path + '/' + str(inx).zfill(6) + '.jpg', normalize=True)
            inx += 1

