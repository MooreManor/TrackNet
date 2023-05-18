from TrackNet_Three_Frames_Input.LoadBatches import TennisDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from TrackNet_Three_Frames_Input.Models.TrackNet import TrackNet_pt
device = 'cuda'
def pt_categorical_crossentropy(pred, label):
    """
    使用pytorch 来实现 categorical_crossentropy
    """
    # print(-label * torch.log(pred))
    return torch.sum(-label * torch.log(pred))


tennis_dt = TennisDataset(images_path="TrackNet_Three_Frames_Input/training_model3_mine.csv", n_classes=256, input_height=360, input_width=640,
                          output_height=360, output_width=640)

data_loader = DataLoader(tennis_dt, batch_size=2, shuffle=False, num_workers=0)
net = TrackNet_pt(n_classes=256, input_height=360, input_width=640).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1.0)

pbar = tqdm(data_loader,
                 total=len(data_loader),
                 desc='Train',)
epochs = 100
for epoch in range(epochs):
    for step, batch in enumerate(pbar):
        # pbar.set_description(f"No.{step}")
        input, label = batch
        input = input.to(device)
        label = label.to(device)
        pred = net(input)
        loss = pt_categorical_crossentropy(pred, label)
        pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
        loss.backward()
        optimizer.step()



