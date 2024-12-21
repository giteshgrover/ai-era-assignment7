import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from model import Model_1, Model_2, Model_3, Model_4
from datetime import datetime
from utils import get_device, transform_data_to_numpy
from torchsummary import summary
import os
from tqdm import tqdm
from torch.utils.data import Subset
import time
import matplotlib.pyplot as plt

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def train_and_test_model():
     # Setup and get device
    device = get_device()
    print(f"\n[INFO] Using device: {device}")

    # Data loading
    print("[STEP 1/5] Preparing datasets...")

    # First calculate the mean and std of the data needed for normalization
    mean = 0.1307 # Precalculated mean of the MNIST dataset
    std = 0.3081 # Precalculated std of the MNIST dataset
    visualize_data = True
    # If mean or std is not provided, calculate it from the data
    if not mean or not std or visualize_data:
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        data_numpy = transform_data_to_numpy(dataset, dataset.data)
        mean = torch.mean(data_numpy)
        std = torch.std(data_numpy)
        if visualize_data:
            printSampleImages(dataset)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

        # Create indices for the first 50000 samples
    train_indices = range(50000)
    test_indices = range(50000, 60000)
    
    train_dataset = Subset(datasets.MNIST('./data', train=True, download=True, transform=train_transform ), train_indices)
    test_dataset = Subset(datasets.MNIST('./data', train=True, download=True, transform=test_transform), test_indices)
    validation_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if (device.type == 'cuda' or device.type == 'mps') else dict(shuffle=True, batch_size=64)
    print(f"[INFO] Dataloader arguments: {dataloader_args}")
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **dataloader_args)

    print(f"[INFO] Total training batches: {len(train_loader)}")
    print(f"[INFO] Batch size: {dataloader_args['batch_size']}")
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}\n")
    print(f"[INFO] Validation samples: {len(validation_dataset)}\n")
    
    # Initialize model
    print("[STEP 2/5] Initializing model...")
    model = Model_4().to(device)
    # Print model summary
    model.to('cpu')
    summary(model, input_size=(1, 28, 28))
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Learning rate StepLR (step size was chosen based on the observation and total num of echos). We will replace this with a better learning approach
    # scheduler = StepLR(optimizer, step_size=6, gamma=0.1) 

    # Training loop
    epochs = 15
    print("[STEP 3/5] Starting training and Testing...")
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n[INFO] Training of Epoch {epoch+1} started...")
        train_model(model, train_loader, optimizer, device, epoch)
        training_time = time.time() - start_time
        print(f"[INFO] Training of Epoch {epoch+1} completed in {training_time:.2f} seconds")
        print("[INFO] Evaluating model...")
        # scheduler.step()
        # print("Current learning rate:", scheduler.get_last_lr()[0])
        test_model(model, test_loader, device)

    print("\n[STEP 4/5] Evaluating model against validation...")
    test_model(model, validation_loader, device)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'model_mnist_{timestamp}.pth'
    # torch.save(model.state_dict(), save_path)

    # Plot the training and testing losses
    print("\n[STEP 5/5] Plot the training and testing losses...")
    printLossAndAccuracy(train_losses, test_losses, train_accuracies, test_accuracies)

    return save_path

def printSampleImages(dataset):
    iter_data = iter(dataset)
    image, label = next(iter_data)
    plt.imshow(image.numpy().squeeze(), cmap='gray_r')

    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        image, label = dataset[index]
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(image.numpy().squeeze(), cmap='gray_r')

def printLossAndAccuracy(train_losses, test_losses, train_accuracies, test_accuracies):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accuracies)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accuracies)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()
#Train this epoch
def train_model(model, train_loader, optimizer, device, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when we start our training loop, we should zero out the gradients so that the parameter update is correct.
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(output, target)
        # loss = nn.functional.nll_loss(output, target)
        train_losses.append(loss.cpu().item())

        # Backpropagation (compute the gradient of the loss with respect to the model parameters and update the parameters)
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar every batch
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'accuracy': f'{accuracy:.2f}%'
        })
        train_accuracies.append(100*correct/total)

def test_model(model, test_loader, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            #loss = nn.CrossEntropyLoss()(output, target, reduction='sum')
            loss = nn.functional.nll_loss(output, target, reduction='sum') # sum up batch loss
            running_loss += loss.cpu().item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        running_loss /= len(test_loader.dataset)
        test_losses.append(running_loss)
        final_accuracy = 100. * correct / total
        print(f"Test Accuracy: {final_accuracy:.2f}%")
        test_accuracies.append(final_accuracy)


if __name__ == "__main__":
    train_and_test_model() 