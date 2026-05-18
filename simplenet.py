"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images

LOGGER = logging.getLogger(__name__)


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced, in_planes, 1, bias=False)
        
        nn.init.constant_(self.fc2.weight, -3.0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)

    def forward(self, x):
        return x * (1.0 + self.ca(x))


class PatchGuard(nn.Module):
    def __init__(self, in_features, momentum=0.1, clip_std=3.0, eps=1e-5):
        super(PatchGuard, self).__init__()
        self.momentum = momentum
        self.clip_std = clip_std
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(1, in_features))
        self.register_buffer('running_std',  torch.ones(1, in_features))
        self.register_buffer('initialized',  torch.tensor(False))

    def forward(self, x):
        if not self.training:
            return x

        with torch.no_grad():
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_std  = x.std(dim=0, keepdim=True).clamp(min=self.eps)
            if not self.initialized.item():
                self.running_mean.copy_(batch_mean)
                self.running_std.copy_(batch_std)
                self.initialized.fill_(True)
            else:
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_std.mul_(1 - self.momentum).add_(batch_std  * self.momentum)

        lo = self.running_mean - self.clip_std * self.running_std
        hi = self.running_mean + self.clip_std * self.running_std
        return torch.max(torch.min(x, hi), lo)
class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None, use_patchguard=True):
        super(Discriminator, self).__init__()

        self.use_patchguard = use_patchguard
        if self.use_patchguard:
            self.patchguard = PatchGuard(in_features=in_planes)

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        if self.use_patchguard:
            x = self.patchguard(x)
        x = self.body(x)
        x = self.tail(x)
        return x

    def forward_without_pg_update(self, x):
        """
        fake_feats 专用前向：冻结 PatchGuard 的 EMA 更新。
        同时冻结 discriminator.body 的 BatchNorm1d 运行统计，
        避免 fake 分支污染 BN 统计。
        """
        if self.use_patchguard:
            was_training = self.patchguard.training
            self.patchguard.eval()
            x = self.patchguard(x)
            if was_training:
                self.patchguard.train()

        was_body_training = self.body.training
        self.body.eval()  # 冻结 BN 运行统计
        x = self.body(x)
        x = self.tail(x)

        if was_body_training:
            self.body.train()
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        return x


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        super(SimpleNet, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        embedding_size=None,
        meta_epochs=1,
        aed_meta_epochs=1,
        gan_epochs=1,
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2,
        dsc_hidden=None,
        dsc_margin=.8,
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0,
        proj_layer_type=0,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        cbam_modules = torch.nn.ModuleList([
            CBAM(in_planes=dim, ratio=16, kernel_size=7) for dim in feature_dimensions
        ]).to(self.device)
        self.forward_modules["cbam_modules"] = cbam_modules

        self.cbam_opt = torch.optim.AdamW(
            self.forward_modules["cbam_modules"].parameters(), lr=lr * 0.1
        )
        LOGGER.info(f"成功植入 {len(cbam_modules)} 个特征级 CBAM 模块（仅通道注意力）")

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(), lr
            )

        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension, self.target_embed_dimension,
                pre_proj, proj_layer_type
            )
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr * .1)

        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std

        self.discriminator = Discriminator(
            self.target_embed_dimension, n_layers=dsc_layers,
            hidden=dsc_hidden, use_patchguard=True
        )
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5
        )
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr * .4
        )
        self.cbam_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.cbam_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, (lr * 0.1) * .4
        )
        self.dsc_margin = dsc_margin
        self.lambda_bce = kwargs.get("lambda_bce", 0.1)

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def embed(self, data):
        """
        公共特征提取接口：无论输入来自 DataLoader 还是单个 batch，
        都返回 features 的 list，保证返回类型一致（避免 DataLoader/非 DataLoader 混用时出错）。
        """
        if isinstance(data, torch.utils.data.DataLoader):
            feats_all = []
            with torch.no_grad():
                for image in data:
                    if isinstance(image, dict):
                        image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                    feats = self._embed(input_image)[0]
                    feats_all.append(feats)
            return feats_all

        return [self._embed(data)[0]]

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B_f, L, C = feat.shape
                H_f = int(math.sqrt(L))
                if H_f * H_f != L:
                    raise ValueError(
                        f"骨干层 {i} 输出序列 L={L} 不是完全平方数，请检查是否含有 CLS token。"
                    )
                features[i] = feat.reshape(B_f, H_f, H_f, C).permute(0, 3, 1, 2)

        cbam_modules = self.forward_modules["cbam_modules"]
        if evaluation:
            with torch.no_grad():
                features = [cbam(feat) for cbam, feat in zip(cbam_modules, features)]
        else:
            features = [cbam(feat) for cbam, feat in zip(cbam_modules, features)]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        return features, patch_shapes

    def test(self, training_data, test_data, save_segmentation_images):
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)

        scores = np.squeeze(np.array(scores))
        segmentations = np.array(segmentations)  # (N, H, W)
        masks_gt_arr = np.squeeze(np.array(masks_gt))
        masks_gt_bin = (masks_gt_arr > 0).astype(np.uint8)

        min_scores = scores.min()
        max_scores = scores.max()
        scores = (scores - min_scores) / (max_scores - min_scores + 1e-8)

        n = len(segmentations)
        seg_min = segmentations.reshape(n, -1).min(axis=-1).reshape(n, 1, 1)
        seg_max = segmentations.reshape(n, -1).max(axis=-1).reshape(n, 1, 1)
        segmentations = (segmentations - seg_min) / (seg_max - seg_min + 1e-8)

        anomaly_labels = [x[1] != "good" for x in test_data.dataset.data_to_iterate]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)

        auroc = metrics.compute_imagewise_retrieval_metrics(scores, anomaly_labels)["auroc"]
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt_bin)
        full_pixel_auroc = pixel_scores["auroc"]

        return auroc, full_pixel_auroc

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores + 1e-8)

        try:
            save_dir = self.ckpt_dir if hasattr(self, 'ckpt_dir') else "./"
            npz_path = os.path.join(save_dir, "mvtec_real_scores.npz")
            np.savez(npz_path, scores=scores, labels=labels_gt)
        except Exception as e:
            print(f"保存分数失败: {e}")

        auroc = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            n = len(segmentations)
            min_scores = segmentations.reshape(n, -1).min(axis=-1).reshape(n, 1, 1)
            max_scores = segmentations.reshape(n, -1).max(axis=-1).reshape(n, 1, 1)
            norm_segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-8)

            masks_gt_arr = np.squeeze(np.array(masks_gt))
            masks_gt_bin = (masks_gt_arr > 0).astype(np.uint8)

            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(norm_segmentations, masks_gt_bin)
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(masks_gt_bin, norm_segmentations)
        else:
            full_pixel_auroc = -1
            pro = -1

        return auroc, full_pixel_auroc, pro

    def train(self, training_data, test_data, save_segmentation_images: bool = False):
        state_dict = {}

        def update_state_dict(d):
            state_dict["discriminator"] = OrderedDict(
                {k: v.detach().cpu() for k, v in self.discriminator.state_dict().items()}
            )
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict(
                    {k: v.detach().cpu() for k, v in self.pre_projection.state_dict().items()}
                )
            state_dict["cbam_modules"] = OrderedDict(
                {k: v.detach().cpu()
                 for k, v in self.forward_modules["cbam_modules"].state_dict().items()}
            )

        best_record = None
        for i_mepoch in range(self.meta_epochs):
            self._train_discriminator(training_data)

            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(
                test_data, scores, segmentations, features, labels_gt, masks_gt
            )
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro
                    update_state_dict(state_dict)

            print(
                f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----"
            )

        # Optional: export anomaly heatmaps once after training.
        if save_segmentation_images:
            try:
                scores, segmentations, _features, _labels_gt, _masks_gt = self.predict(test_data)
                self.save_segmentation_images(test_data, segmentations, scores)
            except Exception as e:
                LOGGER.warning(f"Failed to save segmentation images: {e}")

        return best_record

    @staticmethod
    def _minmax_normalize(x, eps=1e-8):
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def _augment_patch(self, patch):
        """
        轻量版 PatchGuard 风格增强（不依赖 cv2/noise/albumentations）：
          1) 几何：随机仿射（旋转+缩放+平移）通过 grid_sample
          2) 纹理：平滑随机噪声的阈值 mask（Perlin-like）替换部分区域
          3) 额外扰动：轻量 blur + 随机擦除 + 亮度/对比度抖动
        patch: Tensor[C, H, W]
        """
        device = patch.device
        dtype = patch.dtype
        C, h, w = patch.shape

        max_deg = 20.0
        min_scale, max_scale = 0.85, 1.15
        max_shift = 0.10  # translation in normalized coord approx

        angle = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * (max_deg * math.pi / 180.0)
        scale = min_scale + torch.rand(1, device=device, dtype=dtype) * (max_scale - min_scale)
        tx = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * (2.0 * max_shift)
        ty = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * (2.0 * max_shift)

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.zeros((1, 2, 3), device=device, dtype=dtype)
        theta[0, 0, 0] = scale * cos_a
        theta[0, 0, 1] = -scale * sin_a
        theta[0, 0, 2] = tx
        theta[0, 1, 0] = scale * sin_a
        theta[0, 1, 1] = scale * cos_a
        theta[0, 1, 2] = ty

        grid = F.affine_grid(theta, size=(1, C, h, w), align_corners=False)
        patch = F.grid_sample(
            patch.unsqueeze(0),
            grid,
            align_corners=False,
            padding_mode="border",
        ).squeeze(0)

        noise = torch.randn(1, 1, h, w, device=device, dtype=dtype)
        noise = F.avg_pool2d(noise, kernel_size=7, stride=1, padding=3)
        noise = self._minmax_normalize(noise)
        thr = 0.3 + torch.rand(1, device=device, dtype=dtype) * 0.4
        mask = (noise > thr).float().squeeze(0)  # [1,h,w]，与 patch[C,h,w] 广播对齐

        replacement = torch.randn_like(patch) * 0.5
        patch = patch * (1.0 - mask) + replacement * mask

        if torch.rand(1, device=device) < 0.30:
            k = 5 if min(h, w) >= 5 else 3
            patch = F.avg_pool2d(patch.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)

        if torch.rand(1, device=device) < 0.30:
            erase_ratio = 0.01 + torch.rand(1, device=device, dtype=dtype) * 0.03
            area = erase_ratio * (h * w)
            aspect = 0.3 + torch.rand(1, device=device, dtype=dtype) * 2.7

            eh = int(torch.sqrt(area / aspect).clamp(min=2).item())
            ew = int((eh * aspect).clamp(min=2).item())

            eh = min(eh, h - 1) if h > 2 else h
            ew = min(ew, w - 1) if w > 2 else w

            if eh > 1 and ew > 1:
                y0 = int(torch.randint(0, max(1, h - eh + 1), (1,), device=device).item())
                x0 = int(torch.randint(0, max(1, w - ew + 1), (1,), device=device).item())
                patch[:, y0:y0 + eh, x0:x0 + ew] = torch.randn((C, eh, ew), device=device, dtype=dtype) * 0.7

        if torch.rand(1, device=device) < 0.50:
            gamma = 0.85 + torch.rand(1, device=device, dtype=dtype) * 0.30
            beta = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * 0.2
            patch = patch * gamma + beta

        return patch

    def _generate_foreground_anomalies(self, images, masks):
        fake_images = images.clone()
        B, C, H, W = images.shape

        cy_lo, cy_hi = int(H * 0.25), int(H * 0.75)
        cx_lo, cx_hi = int(W * 0.25), int(W * 0.75)

        for i in range(B):
            use_mask = False

            if masks is not None:
                fg_mask  = masks[i, 0] > 0
                fg_ratio = fg_mask.float().mean().item()
                fg_pixels = int(fg_mask.sum().item()) if fg_mask.numel() > 0 else 0
                if fg_ratio >= 0.005 or fg_pixels > 50:
                    fg_indices = torch.nonzero(fg_mask, as_tuple=False)
                    if len(fg_indices) > 0:
                        use_mask = True

            if use_mask:
                idx = torch.randint(0, len(fg_indices), (1,)).item()
                cy  = fg_indices[idx][0].item()
                cx  = fg_indices[idx][1].item()
            else:
                cy = torch.randint(cy_lo, cy_hi, (1,)).item()
                cx = torch.randint(cx_lo, cx_hi, (1,)).item()

            ph = max(2, torch.randint(int(H * 0.05), int(H * 0.20) + 1, (1,)).item())
            pw = max(2, torch.randint(int(W * 0.05), int(W * 0.20) + 1, (1,)).item())

            y1, y2 = max(0, cy - ph), min(H, cy + ph)
            x1, x2 = max(0, cx - pw), min(W, cx + pw)

            target_idx   = (i + 1) % B
            source_patch = images[target_idx, :, y1:y2, x1:x2]
            if source_patch.numel() == 0 or source_patch.shape[-2] < 4 or source_patch.shape[-1] < 4:
                continue
            fake_patch   = self._augment_patch(source_patch)

            img_min = images[i].min().item()
            img_max = images[i].max().item()
            fake_images[i, :, y1:y2, x1:x2] = torch.clamp(fake_patch, min=img_min, max=img_max)

        return fake_images

    def _train_discriminator(self, input_data):
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        self.forward_modules["cbam_modules"].train()

        i_iter = 0
        LOGGER.info("开始训练判别器（CBAM + EMA PatchGuard）...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss   = []
                all_p_true = []
                all_p_fake = []

                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    self.cbam_opt.zero_grad()
                    if self.train_backbone:
                        self.backbone_opt.zero_grad()

                    i_iter += 1

                    img  = data_item["image"]
                    mask = data_item.get("mask", torch.zeros_like(img[:, :1, :, :]))

                    img  = img.to(torch.float).to(self.device)
                    mask = mask.to(self.device)

                    fake_img = self._generate_foreground_anomalies(img, mask)

                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                    else:
                        true_feats = self._embed(img, evaluation=False)[0]

                    if self.pre_proj > 0:
                        fake_feats_img = self.pre_projection(self._embed(fake_img, evaluation=False)[0])
                    else:
                        fake_feats_img = self._embed(fake_img, evaluation=False)[0]

                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(
                        noise_idxs, num_classes=self.mix_noise
                    ).to(self.device)
                    noise = torch.stack([
                        torch.randn_like(true_feats) * min(
                            self.noise_std * 1.1 ** k, self.noise_std * 3.0
                        )
                        for k in range(self.mix_noise)
                    ], dim=1)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    fake_feats_noise = true_feats + noise

                    fake_feats = torch.cat([fake_feats_img, fake_feats_noise], dim=0)

                    true_scores = self.discriminator(true_feats)
                    fake_scores = self.discriminator.forward_without_pg_update(fake_feats)

                    th     = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores  + th, min=0)

                    self.logger.logger.add_scalar("p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar("p_fake", p_fake, self.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()

                    if self.lambda_bce > 0:
                        bce_true = F.binary_cross_entropy_with_logits(
                            true_scores, torch.zeros_like(true_scores)
                        )
                        bce_fake = F.binary_cross_entropy_with_logits(
                            fake_scores, torch.ones_like(fake_scores)
                        )
                        loss = loss + self.lambda_bce * (bce_true + bce_fake)

                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()

                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    self.cbam_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if self.cos_lr:
                    self.dsc_schl.step()
                    self.cbam_schl.step()

                all_loss   = sum(all_loss)   / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str  = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        _ = self.forward_modules.eval()

        img_paths = []
        scores    = []
        masks     = []
        labels_gt = []
        masks_gt  = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                else:
                    image = data
                _scores, _masks, _feats = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, [], labels_gt, masks_gt

    def _predict(self, images):
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(
                images, provide_patch_shapes=True, evaluation=True
            )
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            assert features.shape[0] == batchsize * scales[0] * scales[1], (
                f"features 第一维={features.shape[0]} 与 batchsize*H*W="
                f"{batchsize}*{scales[0]}*{scales[1]} 不一致，可能存在 patch ordering/stride silent bug"
            )
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [x[2] for x in data.dataset.data_to_iterate]
        mask_paths  = [x[3] for x in data.dataset.data_to_iterate]
        # Export only anomaly samples; cap at 5 images per class.
        anomaly_indices = [
            idx for idx, item in enumerate(data.dataset.data_to_iterate)
            if item[1] != "good"
        ]
        max_images_per_class = 5
        if len(anomaly_indices) > max_images_per_class:
            rng = np.random.RandomState(2026)  # fixed seed for reproducible sampling
            anomaly_indices = sorted(
                rng.choice(anomaly_indices, size=max_images_per_class, replace=False).tolist()
            )
        if len(anomaly_indices) == 0:
            LOGGER.warning("No anomaly samples found; skip segmentation image export.")
            return
        image_paths = [image_paths[i] for i in anomaly_indices]
        mask_paths = [mask_paths[i] for i in anomaly_indices]
        segmentations = [segmentations[i] for i in anomaly_indices]
        scores = [scores[i] for i in anomaly_indices]
        save_dir = os.path.join(self.ckpt_dir, "segmentation_images")
        os.makedirs(save_dir, exist_ok=True)

        def image_transform(image):
            in_std  = np.array(data.dataset.transform_std).reshape(-1, 1, 1)
            in_mean = np.array(data.dataset.transform_mean).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            save_dir, image_paths, segmentations, scores, mask_paths,
            image_transform=image_transform, mask_transform=mask_transform
        )


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x