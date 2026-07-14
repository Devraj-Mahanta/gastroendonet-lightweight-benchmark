import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as _base
from collections import OrderedDict
DATA_DIR = _base.DATA_DIR
SPLIT_FILE = _base.SPLIT_FILE
CLASS_NAMES = _base.CLASS_NAMES
NUM_CLASSES = _base.NUM_CLASSES
CHANNELS = _base.CHANNELS
SEED = _base.SEED
RESULTS_DIR = os.path.join('finetune', 'results')
CHECKPOINTS_DIR = os.path.join('finetune', 'checkpoints')
BASE_IMAGE_SIZE = {'efficientnet_b1': 224, 'mobilenetv3_small': 224}
HIGH_IMAGE_SIZE = {'efficientnet_b1': 240, 'mobilenetv3_small': 256}
NUM_RUNS = 3
BATCH_SIZE = 32
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
INITIAL_LR = 2e-05
MIN_LR = 1e-07
WEIGHT_DECAY = 5e-05
HEAD_WARMUP_EPOCHS = 40
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 0.4
STD_AUG = dict(rotation=0.1, zoom=0.15, brightness=0.15, contrast=0.15, saturation=0.15)
STRONG_AUG = dict(rotation=0.15, zoom=0.2, brightness=0.2, contrast=0.2, saturation=0.2)
FOCAL_GAMMA = 0.5
LABEL_SMOOTHING = 0.05
TTA_N_AUGMENTS = 10
ABLATION_CONFIGS: OrderedDict = OrderedDict([('baseline', dict(description='Existing checkpoint, continued training (same settings)', image_size_key='base', schedule='fixed', mixup_alpha=0.0, cutmix_alpha=0.0, strong_aug=False, rich_head=False, adamw=False, focal_loss=False, init_from='checkpoint')), ('a1_resolution', dict(description='Baseline + higher input resolution', image_size_key='high', schedule='fixed', mixup_alpha=0.0, cutmix_alpha=0.0, strong_aug=False, rich_head=False, adamw=False, focal_loss=False, init_from='checkpoint')), ('a2_cosine', dict(description='A1 + cosine LR decay with linear warm-up', image_size_key='high', schedule='cosine', mixup_alpha=0.0, cutmix_alpha=0.0, strong_aug=False, rich_head=False, adamw=False, focal_loss=False, init_from='checkpoint')), ('a3_mixup', dict(description='A2 + MixUp augmentation (alpha=0.4)', image_size_key='high', schedule='cosine', mixup_alpha=0.4, cutmix_alpha=0.0, strong_aug=False, rich_head=False, adamw=False, focal_loss=False, init_from='checkpoint')), ('a4_augstack', dict(description='A3 + CutMix + stronger spatial/colour augmentation', image_size_key='high', schedule='cosine', mixup_alpha=0.4, cutmix_alpha=0.4, strong_aug=True, rich_head=False, adamw=False, focal_loss=False, init_from='checkpoint')), ('a5_richhead', dict(description='A4 + rich classification head (BN→FC512→FC256)', image_size_key='high', schedule='cosine', mixup_alpha=0.4, cutmix_alpha=0.4, strong_aug=True, rich_head=True, adamw=False, focal_loss=False, init_from='checkpoint')), ('a6_adamw', dict(description='A5 + L2 weight decay on Dense layers (1e-4)', image_size_key='high', schedule='cosine', mixup_alpha=0.4, cutmix_alpha=0.4, strong_aug=True, rich_head=True, adamw=True, focal_loss=False, init_from='checkpoint')), ('full', dict(description='Full pipeline: cosine LR + high-res + strong aug + focal loss', image_size_key='high', schedule='cosine', mixup_alpha=0.0, cutmix_alpha=0.0, strong_aug=True, rich_head=False, adamw=False, focal_loss=True, init_from='checkpoint'))])
