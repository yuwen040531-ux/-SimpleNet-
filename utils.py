import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
from matplotlib import font_manager

LOGGER = logging.getLogger(__name__)

def _setup_matplotlib_chinese():
    """尽量确保 matplotlib 中文可用（避免保存图片中文变方块）。"""
    try:
        name_candidates = [
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        picked = next((name for name in name_candidates if name in available), None)
        if picked is None:
            path_candidates = [
                os.environ.get("SIMPLENET_FONT_PATH", ""),
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]
            for font_path in path_candidates:
                if font_path and os.path.exists(font_path):
                    font_manager.fontManager.addfont(font_path)
                    picked = font_manager.FontProperties(fname=font_path).get_name()
                    break
        if picked is not None:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [picked]
        else:
            plt.rcParams["font.sans-serif"] = name_candidates
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_setup_matplotlib_chinese()


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images with thresholded heatmaps."""
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="生成分割热力图...",
        leave=False,
    ):
        # 1. 加载原图，并保留一份未经 Transform 的原图作为热力图底色
        image_pil = PIL.Image.open(image_path).convert("RGB")
        
        # 获取预测矩阵的宽高
        seg_h, seg_w = segmentation.shape
        # 将原图 Resize 到与预测矩阵相同大小，转换为 numpy 数组作为背景
        raw_bg = np.array(image_pil.resize((seg_w, seg_h)))

        # 原始的 transform 流程（为了兼容原有代码结构）
        image = image_transform(image_pil)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None and mask_path != "-1":
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)

        # ==================== 核心修改区：热力图生成与阈值截断 ====================
        
        # (1) 将异常分数归一化到 [0, 1] 区间
        seg_min, seg_max = segmentation.min(), segmentation.max()
        if seg_max > seg_min:
            seg_norm = (segmentation - seg_min) / (seg_max - seg_min)
        else:
            seg_norm = segmentation
            
        # (2) 生成伪彩色热力图 (使用 JET 色谱)
        cmap = plt.get_cmap('jet')
        # cmap 返回的是 RGBA (0-1)，截取前三个通道 RGB，并放大到 0-255
        heatmap = (cmap(seg_norm)[:, :, :3] * 255).astype(np.uint8)
        
        # (3) 设定阈值与透明度
        threshold = 0.5  # 【关键参数】你可以修改这个阈值（0.0 到 1.0），越大过滤的背景越多
        alpha = 0.5      # 【关键参数】热力图覆盖在原图上的不透明度（0.5 代表一半原图，一半热力图）
        
        # (4) 混合叠加：只在异常分数大于阈值的地方叠加热力图颜色，低于阈值的地方保留原图
        overlay = raw_bg.copy()
        mask_bool = seg_norm > threshold
        overlay[mask_bool] = (alpha * heatmap[mask_bool] + (1 - alpha) * raw_bg[mask_bool]).astype(np.uint8)
        
        # =========================================================================

        # 绘图部分
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        
        # 显示原图背景 (比显示 Transform 后的 Tensor 更直观)
        axes[0].imshow(raw_bg)
        axes[0].set_title("原图")
        axes[0].axis('off') # 关闭坐标轴标尺
        
        if masks_provided:
            axes[1].imshow(mask.transpose(1, 2, 0))
            axes[1].set_title("标注掩码")
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title("异常热力图")
            axes[2].axis('off')
        else:
            axes[1].imshow(overlay)
            axes[1].set_title("异常热力图")
            axes[1].axis('off')

        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename, dpi=200, bbox_inches="tight")
        plt.close(f)




def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        #return torch.device("cuda:{}".format(gpu_ids[0]))
        # 确保它可以灵活识别 GPU 编号
        return torch.device("cuda:{}".format(gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
