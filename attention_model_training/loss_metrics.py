import numpy as np
import torch
from metrics import angular_dist_torch
import torch.nn.functional as F

def gaussian_kernel(l=5, sig=1.):
    """
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    creates 1D gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)

def topk(pred, centers, k=2):
    zn_topk, zn_topk_idk = torch.topk(pred, k=k, dim=1)
    zn_topk_smax = torch.softmax(zn_topk, axis=1)
    angle = torch.sum(zn_topk_smax * centers[zn_topk_idk], dim=1)
    return angle

class MultiLabelLossMetrics:
    def __init__(self, bin_data, kernel_length, device, max_length, filt_postfix):
        self.num_bins = len(bin_data['azimuth_bin_centers'])
        az_bin_centers = torch.tensor(bin_data['azimuth_bin_centers']).type(torch.float32).to(device)
        self.azx, self.azy = torch.cos(az_bin_centers), torch.sin(az_bin_centers)

        # Pad zenith bins with edge values
        num_zenith_padding = (kernel_length - 1)//2
        zenith_centers = np.array(bin_data['zenith_bin_centers'])
        padded_zenith_bins = np.concatenate([
                -zenith_centers[num_zenith_padding-1 : : -1],
                zenith_centers,
                2 * np.pi - zenith_centers[-1 : -num_zenith_padding-1 : -1],
        ])
        
        self.zn_bin_centers = torch.tensor(padded_zenith_bins).type(torch.float32).to(device)
        self.max_length = max_length

        local_weights = np.float32(gaussian_kernel(l=kernel_length, sig=kernel_length/5))

        kernel = torch.nn.Parameter(torch.tensor(local_weights)[None, None, :].to(device), requires_grad=False)
        self.az_conv = torch.nn.Conv1d(1, 1, kernel_size=kernel_length, bias=False,
                                        padding_mode='circular', padding='same', device=device)
        self.az_conv.weight = kernel

        self.zn_conv = torch.nn.Conv1d(1, 1, kernel_size=kernel_length,  bias=False,
                                        padding_mode='zeros', padding=kernel_length-1, device=device)
        self.zn_conv.weight = kernel

        self.filt_postfix = filt_postfix
        self.metric_names = ['ang_dist' + self.filt_postfix]

    def nll(self, pred, target):
        pred = pred.log_softmax(dim=1)
        loss = torch.mean(torch.sum(-target * pred, dim=1))
        return loss

    def smoothed_localbin_loss(self, az_class, zn_class, az_disc_pred, zn_disc_pred):
        with torch.no_grad():
            az_onehot = torch.nn.functional.one_hot(az_class, self.num_bins).unsqueeze(1).type(torch.float32)
            zn_onehot = torch.nn.functional.one_hot(zn_class, self.num_bins).unsqueeze(1).type(torch.float32)
            az_smoothed = self.az_conv(az_onehot).squeeze(1)
            zn_smoothed = self.zn_conv(zn_onehot).squeeze(1)

        loss = self.nll(az_disc_pred, az_smoothed) + self.nll(zn_disc_pred, zn_smoothed)
        return loss

    def calculate_loss_and_metrics(self, y, y_pred, is_validation=False):
        y, sequence_lengths = y
        az_disc_pred, zn_disc_pred = y_pred
        with torch.no_grad():
            zn_class = y[:, 1].type(torch.int64)
            az_class = y[:, 0].type(torch.int64)
            azimuth, zenith = y[:, 2], y[:, 3]            

            az_probs = torch.softmax(az_disc_pred, dim=1)
            azmx, azmy = az_probs @ self.azx, az_probs @ self.azy
            azn = torch.sqrt(azmx**2 + azmy**2)
            az_pred = ( torch.arccos(azmx / azn) * torch.sign(azmy) ) % (np.pi * 2)
            
            zn_pred = topk(zn_disc_pred, self.zn_bin_centers, k=3)

            angular_dist = angular_dist_torch(azimuth, zenith, az_pred, zn_pred)
        
        loss = self.smoothed_localbin_loss(az_class, zn_class, az_disc_pred, zn_disc_pred)

        loss_metrics = {
                    "loss": loss,
                    "metrics": {
                        'ang_dist' + self.filt_postfix: angular_dist,
                    }
               }
        if is_validation:
            loss_metrics['extra_data'] = {"zn_pred": zn_pred, "az_pred": az_pred}
        return loss_metrics
