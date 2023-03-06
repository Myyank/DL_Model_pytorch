import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes,
                         kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        out = F.relu(self.bn1(self.conv1(x)))
        # 2. Go through conv2, bn
        out = self.bn2(self.conv2(out))
        # 3. Combine with shortcut output, and go through relu
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        raise NotImplementedError


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """

        images = F.relu(self.bn1(self.conv1(images)))

        images = self.layer1(images)
        images= self.layer2(images)
        images = self.layer3(images)
        images = self.layer4(images)

        images = F.avg_pool2d(images, 4)
        images = images.view(images.size(0), -1)
        logits = self.linear(images)   
        return logits     
        raise NotImplementedError

    def visualize(self, logdir):
        import matplotlib.pyplot as plt
        from matplotlib import pyplot

        # Get the weights of the first convolutional layer
        weights = self.conv1.weight.data.cpu().numpy()
        # Normalize the filter values to 0-1 for visualization
        f_min, f_max = weights.min(), weights.max()
        weights = (weights - f_min) / (f_max - f_min)
        """Visualize the kernels of the  first Conv. layer"""
        fig, axs = plt.subplots(8, 8, figsize=(10, 10))
        # Plot the filters
        for i in range(weights.shape[0]):
            axs[i//8,i%8].imshow(weights[i,0,:,:],cmap='gray')
            axs[i//8,i%8].set_xticks([])
            axs[i//8,i%8].set_yticks([])
        # fig.suptitle("Kernels of the First Convolutional Layer")
        plt.savefig(logdir+"/kernel.png")
        plt.show()
      

