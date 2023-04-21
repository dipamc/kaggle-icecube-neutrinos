import numpy as np
import torch
from metrics import angular_dist_torch
import torch.nn.functional as F

def angle_to_xyz(azimuth, zenith):
    x = (torch.cos(azimuth) * torch.sin(zenith)).unsqueeze(1)
    y = (torch.sin(azimuth) * torch.sin(zenith)).unsqueeze(1)
    z = torch.cos(zenith).unsqueeze(1)
    return torch.cat([x,y,z], dim=1)

def xyz_to_angle(xyz):
    z_normed = xyz[:, 2] / torch.norm(xyz ,dim=1)
    x_normed = xyz[:, 0] / torch.norm(xyz[:, :2] ,dim=1)
    azimuth = ( torch.arccos(x_normed) * torch.sign(xyz[:, 1]) ) % (np.pi * 2)
    zenith = torch.arccos(z_normed)
    return azimuth, zenith

def vMF_loss(xyz_gt, xyz_pred, eps = 1e-8):
    """  von Mises-Fisher Loss: n_true is unit vector ! """
    kappa = torch.norm(xyz_pred, dim=1)
    logC  = - kappa + torch.log( ( kappa + eps ) / ( 1 - torch.exp( -2 * kappa )+ 2 * eps ) )
    return -( (xyz_gt * xyz_pred).sum(dim=1) + logC ).mean() 

class VonMishesFisherLossMetrics:
    def __init__(self, filt_postfix):
        self.filt_postfix = filt_postfix
        self.metric_names = ['ang_dist' + self.filt_postfix, 'ang_dist_vmf']

    def calculate_loss_and_metrics(self, y, y_pred, is_validation=False):
        with torch.no_grad():
            azimuth, zenith = y[:, 2], y[:, 3]
            xyz_gt = angle_to_xyz(azimuth, zenith)
        xyz_pred = y_pred
        loss =  vMF_loss(xyz_gt, xyz_pred)
        with torch.no_grad():
            azp = y[:, 4]
            az_pred, zn_pred = xyz_to_angle(xyz_pred)
            angular_dist = angular_dist_torch(azimuth, zenith, azp, zn_pred)
            angular_dist_vmf = angular_dist_torch(azimuth, zenith, az_pred, zn_pred)
        
        loss_metrics = {
                    "loss": loss, 
                    "metrics": {
                        'ang_dist' + self.filt_postfix: (angular_dist - 1) * 1000,
                        'ang_dist_vmf': (angular_dist_vmf - 1) * 1000,
                    }
            }
        if is_validation:
            loss_metrics['extra_data'] = {"zn_pred": zn_pred, "az_pred": az_pred}
        return loss_metrics