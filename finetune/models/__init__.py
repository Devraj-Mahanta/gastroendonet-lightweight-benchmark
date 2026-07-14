import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune.models.efficientnet_b1_ft import get_model as _b1_ft
from finetune.models.mobilenetv3_small_ft import get_model as _mv3s_ft
FT_REGISTRY = {'efficientnet_b1': _b1_ft, 'mobilenetv3_small': _mv3s_ft}
FT_DISPLAY_NAMES = {'efficientnet_b1': 'EfficientNet-B1 (fine-tuned)', 'mobilenetv3_small': 'MobileNetV3-Small (fine-tuned)'}
PRETRAINED_CKPTS = {'efficientnet_b1': os.path.join('checkpoints', 'efficientnet_b1_run{run}_best.h5'), 'mobilenetv3_small': os.path.join('checkpoints', 'mobilenetv3_small_run{run}_best.h5')}
__all__ = ['FT_REGISTRY', 'FT_DISPLAY_NAMES', 'PRETRAINED_CKPTS']
