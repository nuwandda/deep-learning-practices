import os
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from matplotlib import pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from model import Model


def plot_samples(dataloader, device, title='Images'):
    figsize = (16, 16)
    sample_data = next(iter(dataloader))[0].to(device)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        sample_data,
        padding=2,
        normalize=True
    ).cpu(), (1, 2, 0)))

def plot_class(dataloader, mclass, title='Images', num=64):
    figsize = (16, 16)
    ret = []
    for data in dataloader.dataset:
        if data[1] == mclass:
            ret.append(data[0])
            if len(ret) == num:
                break

    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        ret,
        padding=2,
        normalize=True
    ).cpu(), (1, 2, 0)))

def main():
    def train(epoch, print_every=50):
        # Training procedure
        total_loss = 0
        start_time = time()
        accuracy = []

        for i, batch in enumerate(train_dataloader, 1):
            # Batch of the image from train DataLoader
            minput = batch[0].to(device)
            # Classes that represents cats, dogs or pandas
            target = batch[1].to(device)
            # The output from our model
            moutput = model(minput)

            # Compute Cross Entropy Loss
            loss = criterion(moutput, target)
            total_loss +=  loss.item()

            # For a pure back-propagation, clean the gradients
            optimizer.zero_grad()
            # Back propagate
            loss.backward()
            # Update model parameters
            optimizer.step()

            # Get the index of the maximum prediction
            argmax = moutput.argmax(dim=1)
            # Calculate the accuracy by comparing
            accuracy.append((target==argmax).sum().item() / target.shape[0])

            if i%print_every == 0:
                print('Epoch: [{}]/({}/{}), Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
                    epoch, i, len(train_dataloader), loss.item(), sum(accuracy) / len(accuracy), time() - start_time
                ))

        # Return average training loss
        return  total_loss / len(train_dataloader)

    def test(epoch):
        total_loss = 0
        start_time = time()
        accuracy = []

        # Disable gradient calculations
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                # Batch of the image from train DataLoader
                minput = batch[0].to(device)
                # Classes that represents cats, dogs or pandas
                target = batch[1].to(device)
                # The output from our model
                moutput = model(minput)

                # Compute Cross Entropy Loss
                loss = criterion(moutput, target)
                total_loss += loss.item()

                # Apply SoftMax on model output to get probabilities
                # Get the class with the maximum score
                argmax = moutput.argmax(dim=1)
                # Calculate the accuracy by comparing
                accuracy.append((target == argmax).sum().item() / target.shape[0])

        print('Epoch: [{}], Test Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
            epoch, total_loss / len(test_dataloader), sum(accuracy) / len(accuracy), time() - start_time
        ))
        # Return average testing loss
        return total_loss / len(test_dataloader)

    # Decides the running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using:', device)

    path = '../datasets/animals-10/raw-img'
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    # Transformation function to be applied on images
    transform = transforms.Compose([
        # Flips the images with a probability of 30%
        transforms.RandomHorizontalFlip(p=0.3),
        # Rotates the images at an angle between -40 t0 40
        transforms.RandomRotation(degrees=40),
        # Resizes the images to 300 pixels
        transforms.Resize(300),
        # Crops the center of the image by 256x256
        transforms.CenterCrop(256),
        # Converts to Tensor
        transforms.ToTensor(),
        # Normalizes the Tensor with ImageNet mean and std
        transforms.Normalize(mean=mean, std=std)
    ])

    # With applying the transformations above, create the dataset
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    # Split the dataset into train and test
    train_set, test_set = torch.utils.data.random_split(dataset, (21000, 5179))

    # Create a DataLoader using training dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Create a DataLoader using test dataset
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    model = Model(device).to(device)
    # Visualization for our model
    # summary(model, (3, 256, 256))

    # Using Adam optimizer with 0.0001 learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # And with Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Start training
    test(0)
    train_loss = []
    test_loss = []

    for epoch in range(1, 51):
        train_loss.append(train(epoch, 200))
        test_loss.append(test(epoch))
        print('\n')

        if epoch % 10 == 0:
            torch.save(model, 'model_' + str(epoch) + '.pth')

    # Evaluate the model
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'g', label='Training Loss')
    plt.plot(range(1, len(test_loss) + 1), test_loss, 'b', label='Testing Loss')

    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()