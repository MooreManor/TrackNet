# from TrackNet_Three_Frames_Input.LoadBatches import TennisDataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch
# from TrackNet_Three_Frames_Input.Models.TrackNet import TrackNet_pt
# device = 'cuda'
# def pt_categorical_crossentropy(pred, label):
#     """
#     使用pytorch 来实现 categorical_crossentropy
#     """
#     # print(-label * torch.log(pred))
#     return torch.sum(-label * torch.log(pred))
#
# #
# tennis_dt = TennisDataset(images_path="TrackNet_Three_Frames_Input/training_model3_mine.csv", n_classes=256, input_height=360, input_width=640,
#                           output_height=360, output_width=640)
#
# data_loader = DataLoader(tennis_dt, batch_size=2, shuffle=False, num_workers=0)
# net = TrackNet_pt(n_classes=256, input_height=360, input_width=640).to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=1.0)
#
# pbar = tqdm(data_loader,
#                  total=len(data_loader),
#                  desc='Train',)
# epochs = 100
# for epoch in range(epochs):
#     for step, batch in enumerate(pbar):
#         # pbar.set_description(f"No.{step}")
#         input, label = batch
#         input = input.to(device)
#         label = label.to(device)
#         pred = net(input)
#         loss = pt_categorical_crossentropy(pred, label)
#         pbar.set_postfix({"loss": float(loss.cpu().detach().numpy())})
#         loss.backward()
#         optimizer.step()


from TrackNet_Three_Frames_Input.utils.utils import calculate_velocity, add_csv_col, jud_dir, add_text_to_video
import csv
import numpy as np


# 读取csv文件
with open('./TrackNet_Three_Frames_Input/tennis.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # 跳过第一行
    data = []
    for row in reader:
        if row[1]=='None':
            x, y = None, None
        else:
            x, y = float(row[1]), float(row[2])
        data.append([x, y])

start_frame=8
end_frame=len(data)
# 将数据转换为numpy数组
data = np.array(data)
vols = calculate_velocity(data)
y_vol = vols[:, 1]
hit, start = jud_dir(y_vol)
add_text_to_video('./TrackNet_Three_Frames_Input/168113494209423_TrackNet.mp4', './TrackNet_Three_Frames_Input/hit.mp4', hit, start)




