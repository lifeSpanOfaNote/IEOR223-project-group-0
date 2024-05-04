#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
manualSeed = 541
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Root directory for dataset
dataroot = ("Data/hat_result_new/hat_result_new")
edgeroot=  ("Data/hat_result_new/edge_new")

# Batch size during training
batch_size = 256
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


class HatsEdgesDataset(Dataset):
    def __init__(self, dataroot, edgeroot, transform_hat, transform_edge):
        self.dataroot = dataroot
        self.edgeroot = edgeroot
        self.transform_edge = transform_edge
        self.transform_hat = transform_hat
        # 假设原图和边缘图的文件名（不包括扩展名）是匹配的
        self.filenames = [f for f in os.listdir(dataroot) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 构造原图文件路径
        hat_path = os.path.join(self.dataroot, self.filenames[idx])
        # 构造边缘图文件路径，注意更改扩展名为.png
        edge_path = os.path.join(self.edgeroot, self.filenames[idx].replace('.jpg', '.png'))

        # 加载图片
        hat_image = Image.open(hat_path)
        edge_image = Image.open(edge_path)
        # 如果hat_image是灰度图像，转换为RGB
        if hat_image.mode != 'RGB':
            hat_image = hat_image.convert('RGB')

        # 应用转换（如果有）
        hat_image = self.transform_hat(hat_image)
        edge_image = self.transform_edge(edge_image)

        return edge_image, hat_image

transform_hat = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
transform_edge = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5),
                           ])
dataset=HatsEdgesDataset(dataroot=dataroot, edgeroot=edgeroot, transform_hat=transform_hat,transform_edge=transform_edge)
test_dataset=HatsEdgesDataset(dataroot="Data/hat_result_new/hat_result_test", edgeroot="Data/hat_result_new/edge_test", transform_hat=transform_hat,transform_edge=transform_edge)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
edge_test, hat_test=next(iter(test_dataloader))
edge_test=edge_test.to(device)
hat_test=hat_test.to(device)

# # Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)
class Generator(nn.Module):
    def __init__(self, ngpu,  edge_channels=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # 噪声向量的处理层
        self.noise_process = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True)
        )

        # 边缘图像的初始处理层
        self.edge_process = nn.Sequential(
            nn.Conv2d(edge_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        # 主生成器模块，添加了ResNet块
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            ResNetBlock(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            ResNetBlock(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ResNetBlock(ngf * 2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResNetBlock(ngf),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, edge_image):
        noise = noise.view(-1, nz)
        noise_feature = self.noise_process(noise)
        noise_feature = noise_feature.view(-1, ngf, 4, 4)
        edge_feature = self.edge_process(edge_image)
        combined_feature = torch.cat((noise_feature, edge_feature), 1)
        output = self.main(combined_feature)
        return output
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # 总输入通道数是图像的通道数与边缘图像的通道数之和
        input_channels = 4

        self.main = nn.Sequential(
            # 输入层
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),  # 将特征图大小平均池化到1x1
            nn.Sigmoid()
            # 输出一个单一的数值表示真实或假
        )

    def forward(self, img, edge_img):
        # 在判别器内部合并图像和边缘图像
        # 假设img和edge_img具有相同的空间维度（例如，都是64x64或128x128）
        combined_img = torch.cat((img, edge_img), dim=1)  # 确保在通道维度(dim=1)上合并
        output = self.main(combined_img)
        return output

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

# cGAN损失
criterionGAN = nn.BCELoss()
# L1损失
criterionL1 = nn.L1Loss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop



print("Starting Training Loop...")
# For each epoch
for lambda_L1 in [10000,0,50,100]:# L1损失的权重
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_G_z1_saved = []
    D_G_z2_saved = []
    D_saved = []
    D_losses = []
    iters = 0
    for epoch in range(0,99):
        try:
        # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x|edge)) + log(1 - D(G(z|edge)|edge))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Unpack data
                edge_images, real_images = data

                edge_images = edge_images.to(device)

                real_images = real_images.to(device)
                b_size = real_images.size(0)
                real_label = torch.full((b_size,), 1, dtype=torch.float, device=device)
                fake_label = torch.full((b_size,), 0, dtype=torch.float, device=device)

                # Forward pass real batch through D
                output = netD(real_images, edge_images).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterionGAN(output, real_label)
                errD_real.backward()
                D_x = output.mean().item()
                #print("判别器结束")
                ## Train with all-fake batch
                # Generate fake image batch with G
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_images = netG(noise, edge_images)
                #print("生成器结束")
                # Classify all fake batch with D
                output = netD(fake_images.detach(), edge_images).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterionGAN(output, fake_label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                #print("判别生成结束")
                # Update D
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z|edge)|edge)) + λL1(G)
                ###########################
                netG.zero_grad()
                # Fake labels are real for generator cost
                output = netD(fake_images, edge_images).view(-1)
                # Calculate G's loss based on this output
                errG_GAN = criterionGAN(output, real_label)
                # Calculate G's L1 loss
                errG_L1 = criterionL1(fake_images, real_images) * lambda_L1
                # Combine losses
                errG = errG_GAN + errG_L1
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 20 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                D_saved.append(D_x)
                D_G_z1_saved.append(D_G_z1)
                D_G_z2_saved.append(D_G_z2)

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 300 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake_fixed = netG(noise, edge_images).detach().cpu()
                        fake_test=netG(fixed_noise,edge_test).detach().cpu()
                    img_list.append(vutils.make_grid(fake_fixed, padding=2, normalize=True))

                    # Save real images just once for comparison
                    vutils.save_image(real_images, f'Result/Lambda={lambda_L1}/real_samples_epoch_{epoch}_{i}.png', normalize=True)

                    vutils.save_image(edge_images, f'Result/Lambda={lambda_L1}/edge_images_epoch_{epoch}_{i}.png', normalize=True)
                    # Save generated images
                    vutils.save_image(fake_fixed, f'Result/Lambda={lambda_L1}/fake_samples_epoch_{epoch}_{i}.png', normalize=True)

                    # Save test images
                    vutils.save_image(fake_test, f'Result/Lambda={lambda_L1}/fake_test_epoch_{epoch}_{i}.png',
                                      normalize=True)

                iters += 1

                torch.save(netG, f'Models/{lambda_L1}.pth')

                # if (lambda_L1 == 100) and (epoch % 5 == 0) and (i>=60):
                #     with torch.no_grad():
                #         fake_fixed = netG(noise, edge_images).detach().cpu()
                #
                #     save_dir_fake = f'Result/Detailed_Result/{epoch}/fake'
                #     save_dir_real = f'Result/Detailed_Result/{epoch}/real'
                #
                #     for idx, img in enumerate(fake_fixed):
                #         # torchvision.utils.save_image要求输入是一个tensor或包含tensor的列表
                #         # 我们通过img.unsqueeze(0)来增加一个批次维度，使其变为[1, channels, height, width]
                #         img_path = os.path.join(save_dir_fake, f'image_{iter}.jpg')
                #         vutils.save_image(img.unsqueeze(0), img_path)
                #
                #     for idx, img in enumerate(real_images):
                #         # torchvision.utils.save_image要求输入是一个tensor或包含tensor的列表
                #         # 我们通过img.unsqueeze(0)来增加一个批次维度，使其变为[1, channels, height, width]
                #         img_path = os.path.join(save_dir_real, f'image_{iter}.jpg')
                #         vutils.save_image(img.unsqueeze(0), img_path)

        except RuntimeError as e:
            if "an illegal memory access was encountered" in str(e):
                print(f"在Epoch {epoch} 遇到CUDA错误，跳过此Epoch...")
                continue  # 跳过当前epoch的剩余部分
            else:
                raise  # 如果错误与预期不符，重新抛出异常



    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("D(x) and D(G(z)) during training")
    plt.plot(D_saved, label="D(x)")
    plt.plot(D_G_z2_saved, label="D(G(z))")
    plt.xlabel("iterations")
    plt.ylabel("accuracy rate")
    plt.legend()
    plt.show()