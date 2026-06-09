import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import device, rtmodel_weights
from .rt_model import RTdetector
from .timer_utils import timer

def make_sliding_windows(tensor, window_size):
    with timer(f"make_sliding_windows(window_size={window_size})"):
        n, d = tensor.size()
        windows = []
        for i in range(n - window_size + 1):
            windows.append(tensor[i:i + window_size])
        return torch.stack(windows)

def train_rt_model(triple):
    window_size = 10
    windows = make_sliding_windows(triple, window_size)
    dataset = TensorDataset(windows, windows)

    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    RTmodel = RTdetector(feats=triple.shape[1]).to(device)
    optimizer = torch.optim.AdamW(RTmodel.parameters(), lr=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    n_epochs = 15
    l = torch.nn.MSELoss(reduction="none")

    for epoch in range(n_epochs):
        RTmodel.train()
        losses = []

        with timer(f"RT training epoch {epoch}", cuda_sync=True):
            for batch_x, _ in dataloader:
                batch_x = batch_x.permute(1, 0, 2).to(device)
                elem = batch_x[-1].unsqueeze(0)

                optimizer.zero_grad()
                z = RTmodel(batch_x, elem)

                if isinstance(z, tuple):
                    loss_tensor = (1 / (epoch + 1)) * l(z[0], elem) + (1 - 1 / (epoch + 1)) * l(z[1], elem)
                else:
                    loss_tensor = l(z, elem)

                loss = loss_tensor.mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        scheduler.step()
        print(f"Epoch {epoch} Training Loss: {sum(losses) / len(losses):.6f}")

    torch.save(RTmodel.state_dict(), rtmodel_weights)
    return RTmodel

def RT_test(tripleT):
    with timer("RT_test total", cuda_sync=True):
        window_size = 10
        Twindows = make_sliding_windows(tripleT, window_size)
        Tdataset = TensorDataset(Twindows, Twindows)
        batch_size = 128
        Tdataloader = DataLoader(Tdataset, batch_size=batch_size, shuffle=False, num_workers=0)

        RTmodel = RTdetector(feats=tripleT.shape[1]).to(device)

        with timer("RT_test load_state_dict", cuda_sync=True):
            RTmodel.load_state_dict(torch.load(rtmodel_weights, map_location=device))

        l = torch.nn.MSELoss(reduction="none")
        RTmodel.eval()

        with torch.no_grad():
            all_losses = []
            for batch_x, _ in Tdataloader:
                batch_x = batch_x.permute(1, 0, 2).to(device)
                elem = batch_x[-1].unsqueeze(0)
                z = RTmodel(batch_x, elem)

                if isinstance(z, tuple):
                    z = z[1]

                loss_tensor = l(z, elem)
                all_losses.append(loss_tensor.cpu())

            all_losses = torch.cat(all_losses, dim=1).squeeze(0)

    return all_losses
