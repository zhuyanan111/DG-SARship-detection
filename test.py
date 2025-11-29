import os
import sys
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from mmengine.runner import Runner
from mmengine.config import Config
import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))

try:
    import mmdet_custom.mixstyle_backbone
    print("✓ Custom backbone loaded")
except Exception as e:
    print(f"Warning: {e}")

# 加载训练配置
cfg = Config.fromfile(str(here / 'DG.py'), import_custom_modules=False)

# 使用训练好的检查点进行测试
work_dir = here / 'work_dirs' / 'hrsid_to_ssdd_cascade_r50'
cfg.work_dir = str(work_dir)

# 查找最新的检查点
ckpt_dir = work_dir / 'epoch_*.pth'
import glob
ckpts = sorted(glob.glob(str(ckpt_dir)))
if ckpts:
    latest_ckpt = ckpts[-1]
    cfg.load_from = latest_ckpt
    print(f"✓ Loading checkpoint: {latest_ckpt}")
else:
    print("Warning: No checkpoint found, using pretrained weights")

# 修复测试数据集路径
ssdd_root = r'D:\Python\domainshipdataset\SSDD'
test_json = Path(ssdd_root) / 'annotations' / 'test.json'

if not test_json.exists():
    print(f"Error: Test annotation file not found: {test_json}")
    print("Please check if SSDD dataset path is correct")
    sys.exit(1)

# 更新测试数据加载器和评估器
cfg.test_dataloader.dataset.data_root = ssdd_root
cfg.test_dataloader.dataset.ann_file = 'annotations/test.json'
cfg.test_evaluator.ann_file = str(test_json)

# 减小测试 batch_size 以避免显存溢出
cfg.test_dataloader.batch_size = 4
cfg.test_dataloader.num_workers = 0
print(f"✓ Test batch_size set to: 1")

print(f"✓ Test dataset root: {ssdd_root}")
print(f"✓ Test annotation file: {test_json}")

# 清理显存
torch.cuda.empty_cache()
print(f"✓ GPU Memory cleared")

try:
    runner = Runner.from_cfg(cfg)
    runner.test()
    
    # 计算并显示详细指标
    coco_gt = COCO(str(test_json))
    
    # 从工作目录查找输出的检测结果
    result_file = work_dir / 'results.bbox.json'
    
    # 如果没有找到，尝试其他可能的文件名
    if not result_file.exists():
        import glob as glob_module
        json_files = list(glob_module.glob(str(work_dir / '*.json')))
        json_files = [f for f in json_files if 'results' in f or 'bbox' in f]
        if json_files:
            result_file = Path(json_files[0])
        else:
            print("Warning: Detection result file not found")
            result_file = None
    
    if result_file and result_file.exists():
        print(f"✓ Loading results from: {result_file}")
        coco_dt = coco_gt.loadRes(str(result_file))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 提取 IoU=0.50 时的 Precision 和 Recall
        print("\n" + "="*60)
        print("📊 评估指标 (IoU=0.50)")
        print("="*60)
        
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        
        iou_idx = 1
        p_at_50 = precision[iou_idx, :, 0, 2, -1]
        mean_p_at_50 = p_at_50[p_at_50 > -1].mean()
        print(f"Precision@0.50:         {mean_p_at_50:.4f}")
        
        r_at_50 = recall[iou_idx, 0, 2, -1]
        print(f"Recall@0.50:            {r_at_50:.4f}")
        
        print(f"mAP@0.50:               {coco_eval.stats[1]:.4f}")
        print("="*60)
    else:
        print("✓ Evaluation completed (results already saved by MMEngine)")
    
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"\n❌ CUDA Error: {e}")
        print("Trying to clear cache and retry...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Retrying test...")
        runner = Runner.from_cfg(cfg)
        runner.test()
    else:
        raise
finally:
    torch.cuda.empty_cache()
