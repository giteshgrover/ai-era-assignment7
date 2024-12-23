import torch
import matplotlib.pyplot as plt

def get_device():
    SEED = 1 # Seed is to generate the same random data for each run
    # For reproducibility
    torch.manual_seed(SEED)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}\n")
        torch.cuda.manual_seed(SEED)
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    else:
        torch.mps.manual_seed(SEED)

    return device

def transform_data_to_numpy(dataset, data):
    exp_data = dataset.transform(data.numpy())
    print('[Train]')
    print(' - Numpy Shape:', exp_data.cpu().numpy().shape)
    print(' - Tensor Shape:', exp_data.size())
    print(' - min:', torch.min(exp_data))
    print(' - max:', torch.max(exp_data))
    print(' - mean:', torch.mean(exp_data))
    print(' - std:', torch.std(exp_data))
    print(' - var:', torch.var(exp_data))
    
    return exp_data

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