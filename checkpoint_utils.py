import os
MIN_CHECKPOINT_BYTES = 1024 * 1024
_EFFICIENTNET_BACKBONES = {'efficientnet_b0': 'efficientnetb0', 'efficientnet_b1': 'efficientnetb1'}

def checkpoint_looks_complete(model_name: str, path: str):
    if not os.path.exists(path):
        return (False, 'not found')
    size = os.path.getsize(path)
    if size < MIN_CHECKPOINT_BYTES:
        return (False, f'incomplete file ({size} bytes)')
    backbone_group = _EFFICIENTNET_BACKBONES.get(model_name)
    if backbone_group is None:
        return (True, None)
    try:
        import h5py
        with h5py.File(path, 'r') as h5:
            group = h5.get(backbone_group)
            if group is None and 'model_weights' in h5:
                group = h5['model_weights'].get(backbone_group)
            if group is None:
                return (False, 'missing EfficientNet backbone weights')
            weight_names = group.attrs.get('weight_names', [])
            if len(weight_names) == 0:
                return (False, 'EfficientNet backbone has no saved weights')
    except OSError as exc:
        return (False, f'unreadable H5 file ({exc})')
    return (True, None)
