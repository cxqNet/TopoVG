"""Portable defaults and the existing decoder constants."""
import os

DEFAULT_DATA_ROOT = './Data/RRSIS-D'
DEFAULT_PARSER_JSON = './JsonTree.json'
DEFAULT_SD_MODEL = os.environ.get('TOPOVG_SD_MODEL', './models/stable-diffusion-v1-5')
DEFAULT_QWEN_MODEL = os.environ.get('TOPOVG_QWEN_MODEL', './models/Qwen2.5-7B-Instruct')
PAD_RATIO = 0.007
BETA_C, BETA_H, LAMBDA_REG, CONF_ALPHA = 70.0, 5.0, 0.5, 2.0
TOPK_PEAKS, PEAK_THRESH, MIN_PEAK_DIST = 4, 0.9, 10
FEATURE_KEYS = {'semantic_mask_8', 'semantic_mask_16', 'self_64'}
SAVE_IMAGES_DEFAULT = True
