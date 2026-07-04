from ..utils.misc import load_ext


# Lazily load the compiled `vision3d.ext` extension so that importing the model
# does not require the external compiled `vision3d` package. The extension is only
# needed for point-cloud grid subsampling during raw-data preprocessing, never for
# the model forward (2D-2D, or 2D-3D/3D-3D on already-preprocessed samples).
ext_module = None


def _get_ext_module():
    global ext_module
    if ext_module is None:
        ext_module = load_ext("vision3d.ext", ["grid_subsampling"])
    return ext_module


def grid_subsample_pack_mode(points, lengths, voxel_size):
    """Grid subsampling in pack mode.

    This function is implemented on CPU.

    Args:
        points (array): points in pack mode. (N, 3)
        lengths (array): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (array): sampled points in pack mode (M, 3)
        s_lengths (array): numbers of sampled points in the batch. (B,)
    """
    s_points, s_lengths = _get_ext_module().grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths
