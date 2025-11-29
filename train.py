import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mmengine.runner import Runner
from mmengine.config import Config
import traceback

cfg_path = 'DG.py'
print('CWD:', os.getcwd())
print('Attempting to load config from:', os.path.abspath(cfg_path))

if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"Config file not found: {os.path.abspath(cfg_path)}")

try:
    cfg = Config.fromfile(cfg_path)
    print('Config loaded successfully')
except Exception as e:
    print('Failed to load config file DG.py:')
    traceback.print_exc()
    sys.exit(1)

cfg.work_dir = './work_dirs/DG'
os.makedirs(cfg.work_dir, exist_ok=True)

if 'default_scope' not in cfg:
    cfg.default_scope = 'mmdet'

print(f'Using config: {cfg}')

try:
    runner = Runner.from_cfg(cfg)
    print('Runner created successfully')
    runner.train()
except Exception as e:
    print(f'Training failed with error: {type(e).__name__}: {str(e)}')
    traceback.print_exc()
    sys.exit(1)