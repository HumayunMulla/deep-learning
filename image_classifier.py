import torch

# Check if there is CUDA on the machine or no
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

# import CIFAR-10 Dataset
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print ("Loading Data for Training")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

print ("Loading Data for Testing")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# classes of the images
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Randomly select 4 images and display them with label
import matplotlib.pyplot as plt
import numpy as np

# imshow - Function to show the image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize for display
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Randomly select images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Display the images selected 
imshow(torchvision.utils.make_grid(images))

# Print the labels of the randomly selected images
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))