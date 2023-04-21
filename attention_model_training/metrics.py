import torch
import numpy as np

def angular_dist_torch(az_true, zen_true, az_pred, zen_pred, mean=True):
    az_pred_clipped = torch.clip(az_pred, 0, np.pi * 2)
    zen_pred_clipped = torch.clip(zen_pred, 0, np.pi)
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)
    
    sa2 = torch.sin(az_pred_clipped)
    ca2 = torch.cos(az_pred_clipped)
    sz2 = torch.sin(zen_pred_clipped)
    cz2 = torch.cos(zen_pred_clipped)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    scalar_prod = torch.clip(scalar_prod, min=-1, max=1)
    
    if mean:
        return torch.mean(torch.abs(torch.arccos(scalar_prod)))
    else:
        return torch.abs(torch.arccos(scalar_prod))

def angular_dist_score(az_true, zen_true, az_pred, zen_pred, mean=False):
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod))) if mean else np.abs(np.arccos(scalar_prod))