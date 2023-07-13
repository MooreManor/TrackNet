import torch

def get_heatmap_preds(batch_heatmaps, normalize_keypoints=False):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, height, width])
    '''
    assert batch_heatmaps.ndim == 3, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    height = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[2]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, -1))

    maxvals, idx = torch.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((batch_size, 1))
    idx = idx.reshape((batch_size, 1))

    preds = idx.repeat(1, 2).float()

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = torch.floor((preds[:, 1]) / width)

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 2)
    pred_mask = pred_mask.float()

    preds *= pred_mask

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        preds[:, :, 0] = (preds[:, :, 0] / (width - 1) * 2 - 1)
        preds[:, :, 1] = (preds[:, :, 1] / (height - 1) * 2 - 1)

    return preds, maxvals