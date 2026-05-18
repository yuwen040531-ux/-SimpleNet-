import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
from matplotlib import font_manager


def _setup_matplotlib_chinese():
    """尽量确保 matplotlib 中文可用（避免保存图片中文变方块）。"""
    try:
        candidates = [
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
        picked = next((name for name in candidates if name in available), None)
        plt.rcParams["font.sans-serif"] = [picked] if picked is not None else candidates
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_setup_matplotlib_chinese()

# ================= 配置区域 =================
BASE_DIR = "/root/autodl-tmp/results_mvtec/MVTecAD_Results_FullPower/simplenet_mvtec/run_sota/models/0"
SAVE_PATH = "./all_mvtec_real_anomaly_dist.jpg" # 保存的大图名称
# ===========================================

def generate_combined_histograms():
    if not os.path.exists(BASE_DIR):
        print(f"❌ 错误：找不到基础路径 {BASE_DIR}！")
        return

    # 1. 动态查找所有的 npz 文件
    search_pattern = os.path.join(BASE_DIR, "*", "mvtec_real_scores.npz")
    npz_files = glob.glob(search_pattern)

    num_files = len(npz_files)
    if num_files == 0:
        print("❌ 错误：没有找到任何 mvtec_real_scores.npz 文件。")
        return

    print(f"🔍 找到了 {num_files} 个类别的分数文件，准备绘制在一张总图上...")

    # 2. 设置子图网格布局 (默认 5 列，自动计算需要的行数)
    cols = 5
    rows = math.ceil(num_files / cols)
    
    # 创建超大画布 (每一列宽4，每一行高3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes = axes.flatten() # 将二维矩阵展平，方便我们用一维循环遍历

    # 3. 循环处理每一个类别并画在对应的子图(ax)上
    for i, file_path in enumerate(npz_files):
        ax = axes[i] # 获取当前的子图区域
        
        folder_name = os.path.basename(os.path.dirname(file_path)) 
        category_name = folder_name.replace("mvtec_", "") 
        print(f"[{category_name.capitalize()}] 正在绘制...")

        # 读取数据
        data = np.load(file_path)
        all_scores = data['scores']
        all_labels = data['labels']

        # 分离数据
        normal_scores = all_scores[all_labels == 0]
        anomaly_scores = all_scores[all_labels == 1]

        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            ax.set_title(f"{category_name.capitalize()} (Data Missing)", fontsize=12)
            continue

        # 归一化
        min_score = min(np.min(normal_scores), np.min(anomaly_scores))
        max_score = max(np.max(normal_scores), np.max(anomaly_scores))
        if max_score > min_score:
            normal_scores = (normal_scores - min_score) / (max_score - min_score)
            anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)

        # 绘制直方图
        ax.hist(normal_scores, bins=50, alpha=0.6, color='#3498db', label='正常', edgecolor='white', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color='#e74c3c', label='异常', edgecolor='white', density=True)

        # 绘制 KDE 密度曲线
        from scipy.stats import gaussian_kde
        x_range = np.linspace(0, 1, 200)
        try:
            ax.plot(x_range, gaussian_kde(normal_scores)(x_range), color='#2980b9', lw=2)
            ax.plot(x_range, gaussian_kde(anomaly_scores)(x_range), color='#c0392b', lw=2)
        except:
            pass

        # 装饰当前子图
        ax.set_title(f"{category_name.capitalize()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("归一化分数", fontsize=10)
        ax.set_ylabel("密度", fontsize=10)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2) # 调整图例位置防止遮挡
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    # 4. 隐藏多余的空白子图 (如果类别数不是5的倍数)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 添加全局超级标题 (可选)
    fig.suptitle("MVTec AD 各类别异常分数分布统计", fontsize=22, y=1.02, fontweight='bold')

    # 5. 调整子图间距并保存
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✨ 大功告成！全家福组合分布图已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    generate_combined_histograms()
