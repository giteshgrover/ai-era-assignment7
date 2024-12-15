import torch

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

def transform_data_to_numpy(dataset):
    exp_data = dataset.train_data
    exp_data = dataset.transform(exp_data.numpy())
    print('[Train]')
    print(' - Numpy Shape:', exp.train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', exp.train_data.size())
    print(' - min:', torch.min(exp_data))
    print(' - max:', torch.max(exp_data))
    print(' - mean:', torch.mean(exp_data))
    print(' - std:', torch.std(exp_data))
    print(' - var:', torch.var(exp_data))
    
    return exp_data