import torch


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def rgb_to_lab(srgb):
    """sRGB [..., 3] in 0..1 -> CIE Lab (D65). The result stays on the compute device (cuda/mps/cpu)."""
    device = _device()
    srgb_pixels = torch.reshape(srgb, [-1, 3]).to(device)

    linear_mask = (srgb_pixels <= 0.04045).float()
    exponential_mask = (srgb_pixels > 0.04045).float()
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ], dtype=torch.float32, device=device)

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
    xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754], dtype=torch.float32, device=device))

    epsilon = 6.0/29.0
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).float()
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).float()
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask

    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0],  # fx
        [116.0, -500.0,  200.0],  # fy
        [  0.0,    0.0, -200.0],  # fz
    ], dtype=torch.float32, device=device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0], dtype=torch.float32, device=device)

    return torch.reshape(lab_pixels, srgb.shape)


def lab_to_rgb(lab):
    """CIE Lab (D65) [..., 3] -> sRGB in 0..1, returned on the input's device."""
    device = _device()
    lab_pixels = torch.reshape(lab, [-1, 3]).to(device)

    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0],  # l
        [1/500.0,     0.0,      0.0],  # a
        [    0.0,     0.0, -1/200.0],  # b
    ], dtype=torch.float32, device=device)
    fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0], dtype=torch.float32, device=device), lab_to_fxfyfz)

    epsilon = 6.0/29.0
    linear_mask = (fxfyfz_pixels <= epsilon).float()
    exponential_mask = (fxfyfz_pixels > epsilon).float()
    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

    # Denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754], dtype=torch.float32, device=device))

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434],  # x
        [-1.5371385,  1.8760108, -0.2040259],  # y
        [-0.4985314,  0.0415560,  1.0572252],  # z
    ], dtype=torch.float32, device=device)
    rgb_pixels = torch.clamp(torch.mm(xyz_pixels, xyz_to_rgb), 0.0, 1.0)

    linear_mask = (rgb_pixels <= 0.0031308).float()
    exponential_mask = (rgb_pixels > 0.0031308).float()
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    return torch.reshape(srgb_pixels.to(lab.device), lab.shape)
