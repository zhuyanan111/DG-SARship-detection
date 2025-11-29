import os
import sys
import traceback
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path
from mmengine.runner import Runner
from mmengine.config import Config
import torch

try:
    here = Path(__file__).resolve().parent
    config_path = here / 'DG.py'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    sys.path.insert(0, str(here))
    
    try:
        import mmdet_custom.mixstyle_backbone
        print("✓ mmdet_custom.mixstyle_backbone imported successfully")
    except Exception as ie:
        print(f"Warning: Error importing MixStyleResNet module: {ie}")

    try:
        cfg = Config.fromfile(str(config_path), import_custom_modules=False)
        print("✓ Config loaded successfully with base inheritance")
    except TypeError as te:
        print(f"Config.fromfile TypeError: {te}, falling back to manual load")
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        cfg_dict = {k: v for k, v in config_module.__dict__.items() if not k.startswith('_')}
        
        if 'custom_imports' in cfg_dict:
            print("Removing custom_imports from config to avoid version compatibility issues")
            del cfg_dict['custom_imports']
        
        cfg = Config(cfg_dict)

    has_val_dataloader = 'val_dataloader' in cfg and cfg.val_dataloader is not None
    has_val_cfg = 'val_cfg' in cfg and cfg.val_cfg is not None
    has_val_evaluator = 'val_evaluator' in cfg and cfg.val_evaluator is not None
    
    if has_val_dataloader or has_val_evaluator:
        if not has_val_cfg:
            print("Adding missing val_cfg")
            cfg.val_cfg = dict(type='ValLoop')
    
    has_test_dataloader = 'test_dataloader' in cfg and cfg.test_dataloader is not None
    has_test_cfg = 'test_cfg' in cfg and cfg.test_cfg is not None
    has_test_evaluator = 'test_evaluator' in cfg and cfg.test_evaluator is not None
    
    if has_test_dataloader or has_test_evaluator:
        if not has_test_cfg:
            print("Adding missing test_cfg")
            cfg.test_cfg = dict(type='TestLoop')

    if 'work_dir' not in cfg or not cfg.work_dir:
        cfg.work_dir = str(here / 'work_dirs' / 'DG')
    if not os.path.isabs(cfg.work_dir):
        cfg.work_dir = str(here / cfg.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)

    if 'default_scope' not in cfg:
        cfg.default_scope = 'mmdet'

    # 优化显存使用，增大 batch_size
    print(f"Original batch_size: {cfg.train_dataloader.batch_size}")
    cfg.train_dataloader.batch_size = 4  # 从 8 改回 4（更稳定）
    cfg.val_dataloader.batch_size = 2
    cfg.test_dataloader.batch_size = 1
    print(f"Adjusted batch_size - train: 4, val: 2, test: 1")
    
    # 调整学习率（batch_size 增大，学习率也要增大）
    if hasattr(cfg, 'optim_wrapper') and 'optimizer' in cfg.optim_wrapper:
        original_lr = cfg.optim_wrapper.optimizer.get('lr', 0.01)
        new_lr = original_lr * (4 / 2)  # 改为 4/2 而不是 8/2
        cfg.optim_wrapper.optimizer['lr'] = new_lr
        print(f"Adjusted learning rate: {original_lr} -> {new_lr}")

    # 自动查找并加载最新的检查点
    work_dir = Path(cfg.work_dir)
    ckpt_pattern = str(work_dir / 'epoch_*.pth')
    ckpts = sorted(glob.glob(ckpt_pattern), key=lambda x: int(x.split('epoch_')[1].split('.pth')[0]))
    
    if ckpts:
        latest_ckpt = ckpts[-1]
        epoch_num = int(latest_ckpt.split('epoch_')[1].split('.pth')[0])
        cfg.resume = True
        cfg.load_from = latest_ckpt
        print(f"✓ Found checkpoint: {latest_ckpt}")
        print(f"✓ Resuming training from epoch {epoch_num}")
    else:
        print("No checkpoint found, starting fresh training")

    torch.cuda.empty_cache()
    
    # 打印 GPU 信息
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    runner = Runner.from_cfg(cfg)
    runner.train()
except Exception as e:
    error_msg = traceback.format_exc()
    print(error_msg)
    with open(here / 'error.log', 'w') as f:
        f.write(error_msg)
    torch.cuda.empty_cache()
    sys.exit(1)
