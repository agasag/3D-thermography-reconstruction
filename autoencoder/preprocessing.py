from scipy.ndimage import morphology


def source_hole_filling(volume, hole_temp):
    mask = volume > 0
    mask_filled = morphology.binary_fill_holes(mask)

    hole_mask = mask_filled.astype(int) - mask.astype(int)

    volume = volume + hole_mask * hole_temp
    return volume