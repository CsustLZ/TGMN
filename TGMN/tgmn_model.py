import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from vita.vita_model import Vita
from .modeling.tgmn_criterion import TGMNSetCriterion
from .modeling.tgmn_matcher import TGMNHungarianMatcher
from .modeling.tgmn import TGMNVIS


@META_ARCH_REGISTRY.register()
class TGMN(Vita):

    @configurable
    def __init__(
            self,
            len_clip_window: int,
            tgmn_criterion: nn.Module,
            memory_size: int,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.len_clip_window = len_clip_window
        self.tgmn_criterion = tgmn_criterion
        self.freeze_detector = kwargs["freeze_detector"]
        self.K = memory_size

        hidden_dim = kwargs["vita_module"].num_queries
        self.temporal_pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    @classmethod
    def from_config(cls, cfg):
        rets = Vita.from_config(cfg)

        rets["vita_module"] = TGMNVIS(cfg=cfg)

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sim_weight = cfg.MODEL.VITA.SIM_WEIGHT
        gate_weight = cfg.MODEL.GENVIS.GATE_WEIGHT

        tgmn_matcher = TGMNHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        tgmn_weight_dict = {
            "loss_tgmn_ce": class_weight, "loss_tgmn_mask": mask_weight,
            "loss_tgmn_dice": dice_weight, "loss_tgmn_gate": gate_weight,
        }
        if sim_weight > 0.0:
            tgmn_weight_dict["loss_tgmn_sim"] = sim_weight

        if cfg.MODEL.VITA.DEEP_SUPERVISION:
            aux_weight_dict = {}
            for i in range(cfg.MODEL.VITA.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in tgmn_weight_dict.items()})
            tgmn_weight_dict.update(aux_weight_dict)
        tgmn_losses = ["tgmn_labels", "tgmn_masks"]
        if sim_weight > 0.0:
            tgmn_losses.append("fg_sim")

        num_classes = rets["sem_seg_head"].num_classes
        tgmn_criterion = TGMNSetCriterion(
            num_classes,
            matcher=tgmn_matcher,
            weight_dict=tgmn_weight_dict,
            eos_coef=cfg.MODEL.VITA.NO_OBJECT_WEIGHT,
            losses=tgmn_losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            sim_use_clip=cfg.MODEL.VITA.SIM_USE_CLIP,
            loss_gate=cfg.MODEL.GENVIS.GATED_PROP,
        )

        rets.update({
            "len_clip_window": cfg.MODEL.GENVIS.LEN_CLIP_WINDOW,
            "tgmn_criterion": tgmn_criterion,
            "memory_size": 5,
        })

        return rets

    def train_model(self, batched_inputs):
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        BT = len(images)
        T = self.num_frames if self.training else BT
        B = BT // T

        outputs, frame_queries, mask_features = self.sem_seg_head(features)

        L, BT, fQ, C = frame_queries.shape
        del features

        mask_features = self.vita_module.vita_mask_features(mask_features)
        mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])

        frame_targets, video_targets = self.prepare_targets(batched_inputs, images)

        losses, fg_indices = self.criterion(outputs, frame_targets)

        if self.freeze_detector:
            losses = dict()

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        num_clips = T // self.len_clip_window

        frame_targets = self.split_frame_targets(frame_targets, B)
        video_targets = self.split_video_targets(video_targets)
        fg_indices = self.split_fg_indices(fg_indices, B)

        frame_queries = frame_queries.reshape(L, B, T, fQ, C)
        frame_queries = frame_queries.split(self.len_clip_window, dim=2)
        mask_features = mask_features.split(self.len_clip_window, dim=1)

        M = {"k": [], "v": []}

        prev_clip_indices = None
        prev_aux_clip_indices = None
        initial_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, L * B, 1)

        prop_q = initial_q
        prev_state_token = None

        for c_i in range(num_clips):
            clip_targets = video_targets[c_i]
            frame_targets_per_clip = frame_targets[c_i]
            frame_queries_per_clip = frame_queries[c_i]
            mask_features_per_clip = mask_features[c_i]
            fg_indices_per_clip = fg_indices[c_i]

            M_enhanced = self.enhance_memory_with_temporal_pos(M, prop_q, L, B)

            vita_outputs, prop_q, prev_state_token = self.vita_module(
                frame_queries_per_clip.flatten(1, 2),
                M_enhanced,
                prop_q,
                prev_state_token
            )
            vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"],
                                                      mask_features_per_clip)
            for out in vita_outputs["aux_outputs"]:
                out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], mask_features_per_clip)

            tgmn_loss_dict, out_clip_indices, aux_clip_indices_list = self.tgmn_criterion(
                vita_outputs,
                clip_targets,
                frame_targets_per_clip,
                fg_indices_per_clip,
                prev_clip_indices,
                prev_aux_clip_indices
            )
            tgmn_weight_dict = self.tgmn_criterion.weight_dict

            loss_dict_keys = list(tgmn_loss_dict.keys())
            for k in loss_dict_keys:
                if k in tgmn_weight_dict:
                    tgmn_loss = tgmn_loss_dict.pop(k)
                    tgmn_loss_dict[f"{k}_clip{c_i}"] = tgmn_loss * tgmn_weight_dict[k]
            losses.update(tgmn_loss_dict)

            self.update_memory(M, vita_outputs["pre_memory"], prop_q)

            prev_clip_indices = out_clip_indices
            prev_aux_clip_indices = aux_clip_indices_list
        return losses

    def update_memory(self, M, new_memory, current_q):
        new_k = new_memory["k"]
        new_v = new_memory["v"]

        if len(M["k"]) < self.K:
            M["k"].append(new_k)
            M["v"].append(new_v)
        else:
            L, B, cQ, C = new_k.shape
            current_q_flat = current_q.view(L * B, cQ, C).mean(dim=0)

            min_sim_idx = -1
            min_sim = float('inf')

            for idx in range(len(M["k"])):
                mem_k = M["k"][idx].view(L * B, cQ, C).mean(dim=0)
                sim = F.cosine_similarity(current_q_flat.flatten(), mem_k.flatten(), dim=0)
                if sim < min_sim:
                    min_sim = sim
                    min_sim_idx = idx

            M["k"][min_sim_idx] = new_k
            M["v"][min_sim_idx] = new_v

    def enhance_memory_with_temporal_pos(self, M, current_q, L, B):
        if len(M["k"]) == 0:
            return M

        M_enhanced = {"k": [], "v": []}

        cQ, LB, C = current_q.shape
        current_q_reshaped = current_q.view(cQ, L, B, C)

        for idx in range(len(M["k"])):
            mem_k = M["k"][idx]
            mem_v = M["v"][idx]

            L_m, B_m, cQ_m, C_m = mem_k.shape

            mem_k_reshaped = mem_k.view(L_m, B_m, cQ_m, C_m)
            temporal_offset = current_q_reshaped.mean(dim=2) - mem_k_reshaped.mean(dim=2)

            temporal_pos = self.temporal_pos_mlp(temporal_offset)
            temporal_pos = temporal_pos.unsqueeze(2).expand(-1, -1, cQ_m, -1)

            mem_k_enhanced = mem_k + temporal_pos.reshape(L_m, B_m, cQ_m, C_m)

            M_enhanced["k"].append(mem_k_enhanced)
            M_enhanced["v"].append(mem_v)

        return M_enhanced

    def split_frame_targets(self, frame_targets, batch_size):
        T = self.num_frames
        W = self.len_clip_window
        num_clips = T // W

        frame_targets = [frame_targets[b_i * T:(b_i + 1) * T] for b_i in range(batch_size)]

        frame_targets_splits = dict()
        for frame_targets_per_batch in frame_targets:
            for clip_idx in range(num_clips):
                if not clip_idx in frame_targets_splits:
                    frame_targets_splits[clip_idx] = []

                frame_targets_splits[clip_idx] += frame_targets_per_batch[clip_idx * W:(clip_idx + 1) * W]

        return list(frame_targets_splits.values())

    def split_video_targets(self, clip_targets):
        clip_len = self.len_clip_window

        clip_target_splits = dict()
        for targets_per_video in clip_targets:
            labels = targets_per_video["labels"]
            ids = targets_per_video["ids"]
            masks = targets_per_video["masks"]
            frame_idx = targets_per_video["frame_idx"]

            masks_splits = masks.split(clip_len, dim=1)
            ids_splits = ids.split(clip_len, dim=1)

            prev_valid = torch.zeros_like(labels).bool()
            for clip_idx, (_masks, _ids) in enumerate(zip(masks_splits, ids_splits)):
                valid_inst = _masks.sum(dim=(1, 2, 3)) > 0.
                new_inst = (prev_valid == False) & (valid_inst == True)

                if not clip_idx in clip_target_splits:
                    clip_target_splits[clip_idx] = []

                prev_valid = prev_valid | valid_inst

                clip_target_splits[clip_idx].append(
                    {
                        "labels": labels, "ids": _ids, "masks": _masks,
                        "video_len": targets_per_video["video_len"],
                        "frame_idx": frame_idx[clip_idx * clip_len:(clip_idx + 1) * clip_len],
                        "valid_inst": valid_inst,
                        "new_inst": new_inst,
                        "prop_inst": prev_valid,
                    }
                )

                prev_valid = prev_valid | valid_inst

        return list(clip_target_splits.values())

    def split_fg_indices(self, fg_indices, batch_size):
        L = len(fg_indices)
        T = self.num_frames
        W = self.len_clip_window
        num_clips = T // W

        fg_indices_splits = []
        for L_i in range(L):
            fg_indices_Li = [fg_indices[L_i][b_i * T:(b_i + 1) * T] for b_i in range(batch_size)]
            fg_indices_Li_splits = dict()
            for b_i in range(batch_size):
                for clip_idx in range(num_clips):
                    if not clip_idx in fg_indices_Li_splits:
                        fg_indices_Li_splits[clip_idx] = []
                    fg_indices_Li_splits[clip_idx] += fg_indices_Li[b_i][clip_idx * W:(clip_idx + 1) * W]
            fg_indices_splits.append(fg_indices_Li_splits)

        fg_indices_splits_clips = []
        for clip_idx in range(num_clips):
            per_clip = []
            for L_i in range(L):
                per_clip.append(fg_indices_splits[L_i][clip_idx])
            fg_indices_splits_clips.append(per_clip)

        return fg_indices_splits_clips

    def inference(self, batched_inputs):
        mask_features = []
        num_frames = len(batched_inputs["image"])
        to_store = self.device

        mask_cls, mask_embed = [], []
        M = {"k": [], "v": []}

        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, 1, 1)
        prev_state_token = None

        for i in range(math.ceil(num_frames / self.len_clip_window)):
            images = batched_inputs["image"][i * self.len_clip_window: (i + 1) * self.len_clip_window]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            outputs, frame_queries, _mask_features = self.sem_seg_head(features)

            M_enhanced = self.enhance_memory_with_temporal_pos(M, output_q, 1, 1)

            vita_outputs, output_q, prev_state_token = self.vita_module(
                frame_queries, M_enhanced, output_q, prev_state_token
            )

            _mask_features = self.vita_module.vita_mask_features(_mask_features)
            mask_features.append(_mask_features)

            mask_cls.append(vita_outputs["pred_logits"][-1])
            mask_embed.append(vita_outputs["pred_mask_embed"][-1])

            self.update_memory(M, vita_outputs["pre_memory"], output_q)
            del vita_outputs

        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]

        out_height = batched_inputs.get("height", image_size[0])
        out_width = batched_inputs.get("width", image_size[1])

        del outputs, images, batched_inputs

        mask_cls = torch.cat(mask_cls)
        mask_embed = torch.cat(mask_embed)

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                                                                     1).flatten(0, 1)
        num_topk = self.test_topk_per_image

        scores = F.softmax(mask_cls, dim=-1)[:, :, :-1]
        scores_per_video, _ = scores.max(dim=0)
        scores_per_video, topk_indices = scores_per_video.flatten().topk(num_topk, sorted=False)

        labels_per_video = labels[topk_indices]
        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')

        mask_embed = mask_embed[:, topk_indices]

        masks_per_video = []
        numerator = torch.zeros(len(topk_indices), dtype=torch.float, device=self.device)
        denominator = torch.zeros(len(topk_indices), dtype=torch.float, device=self.device)

        for i in range(math.ceil(num_frames / self.len_clip_window)):
            mask_pred = torch.einsum("qc,tchw->qthw", mask_embed[i], mask_features[i])

            mask_pred = retry_if_cuda_oom(F.interpolate)(
                mask_pred,
                size=interim_size,
                mode="bilinear",
                align_corners=False,
            )

            mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

            interim_mask_soft = mask_pred.sigmoid()
            interim_mask_hard = interim_mask_soft > 0.5

            numerator += (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
            denominator += interim_mask_hard.flatten(1).sum(1)

            mask_pred = F.interpolate(
                mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
            ) > 0.

            masks_per_video.append(mask_pred.to(to_store))

        masks_per_video = torch.cat(masks_per_video, dim=1).cpu()
        scores_per_video *= (numerator / (denominator + 1e-6))

        processed_results = {
            "image_size": (out_height, out_width),
            "pred_scores": scores_per_video.tolist(),
            "pred_labels": labels_per_video.tolist(),
            "pred_masks": masks_per_video,
        }

        return processed_results