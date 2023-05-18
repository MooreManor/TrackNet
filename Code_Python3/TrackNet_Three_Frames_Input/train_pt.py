from LoadBatches import TennisDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from Models.TrackNet import TrackNet_pt
import argparse
# --save_weights_path=weights/model --training_images_name="training_model3_mine.csv" --epochs=100 --n_classes=256 --input_height=360 --input_width=640 --batch_size=2
# --load_weights=2 --step_per_epochs=200
#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 1000 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "-1")
# parser.add_argument("--step_per_epochs", type = int, default = 200 )

args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
# step_per_epochs = args.step_per_epochs

device = 'cuda'
def pt_categorical_crossentropy(pred, label):
    """
    使用pytorch 来实现 categorical_crossentropy
    """
    # print(-label * torch.log(pred))
    return torch.sum(-label * torch.log(pred))


tennis_dt = TennisDataset(images_path=training_images_name, n_classes=n_classes, input_height=input_height, input_width=input_width,
                          output_height=input_height, output_width=input_width)

data_loader = DataLoader(tennis_dt, batch_size=train_batch_size, shuffle=True, num_workers=8)
net = TrackNet_pt(n_classes=n_classes, input_height=input_height, input_width=input_width).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1.0)

pbar = tqdm(data_loader,
                 total=len(data_loader),
                 desc='Train')

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
    if epoch % 5 == 0:
        torch.save(net.state_dict(), save_weights_path + ".0")



