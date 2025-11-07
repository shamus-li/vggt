# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse  # noqa: E402

import pycolmap  # noqa: E402
import trimesh  # noqa: E402

from vggt.dependency.np_to_pycolmap import (  # noqa: E402
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from vggt.dependency.track_predict import predict_tracks  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.helper import (  # noqa: E402
    create_pixel_coordinate_grid,
    randomly_limit_trues,
)
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory containing the scene images",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_ba", action="store_true", default=False, help="Use BA for reconstruction"
    )
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Maximum reprojection error for reconstruction",
    )
    parser.add_argument(
        "--shared_camera",
        action="store_true",
        default=False,
        help="Use shared camera for all images",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        help="Camera type for reconstruction",
    )
    parser.add_argument(
        "--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks"
    )
    parser.add_argument(
        "--query_frame_num", type=int, default=8, help="Number of frames to query"
    )
    parser.add_argument(
        "--max_query_pts", type=int, default=4096, help="Maximum number of query points"
    )
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Use fine tracking (slower but more accurate)",
    )
    parser.add_argument(
        "--conf_thres_value",
        type=float,
        default=5.0,
        help="Confidence threshold value for depth filtering (wo BA)",
    )
    parser.add_argument(
        "--min_inlier_per_frame",
        type=int,
        default=64,
        help="Minimum number of inliers required per frame during bundle adjustment",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["both", "vggt", "ba"],
        help="Pipeline stage to execute: 'vggt' (VGGT only), 'ba' (bundle adjustment only), or 'both'",
    )
    parser.add_argument(
        "--stage_cache",
        type=str,
        default=None,
        help="Optional path to store/load intermediate VGGT outputs when splitting stages",
    )
    return parser.parse_args()


DEFAULT_STAGE_CACHE_NAME = "cache_vggt_result.pt"


def set_random_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Setting seed as: {seed}")


def determine_device_and_dtype():
    if torch.cuda.is_available():
        major_capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_capability >= 8 else torch.float16
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


def get_stage_cache_path(args: argparse.Namespace) -> str:
    if args.stage_cache:
        return args.stage_cache
    return os.path.join(args.scene_dir, DEFAULT_STAGE_CACHE_NAME)


def save_vggt_stage_outputs(vggt_result: "VGGTLoopResult", cache_path: str):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    data = {
        "images": vggt_result.images.detach().cpu(),
        "original_coords": vggt_result.original_coords.detach().cpu(),
        "base_image_path_list": vggt_result.base_image_path_list,
        "extrinsic": vggt_result.extrinsic,
        "intrinsic": vggt_result.intrinsic,
        "depth_map": vggt_result.depth_map,
        "depth_conf": vggt_result.depth_conf,
        "points_3d": vggt_result.points_3d,
        "vggt_fixed_resolution": vggt_result.vggt_fixed_resolution,
        "img_load_resolution": vggt_result.img_load_resolution,
        "pred_tracks": vggt_result.pred_tracks,
        "pred_vis_scores": vggt_result.pred_vis_scores,
        "pred_confs": vggt_result.pred_confs,
        "tracked_points_3d": vggt_result.tracked_points_3d,
        "points_rgb": vggt_result.points_rgb,
    }
    torch.save(data, cache_path)
    print(f"Saved VGGT stage outputs to {cache_path}")


def load_vggt_stage_outputs(args: argparse.Namespace, cache_path: str) -> "VGGTLoopResult":
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Stage cache not found at {cache_path}. Run with --stage vggt first."
        )

    device, dtype = determine_device_and_dtype()
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    data = torch.load(cache_path, map_location="cpu")
    print(f"Loaded VGGT stage outputs from {cache_path}")

    return VGGTLoopResult(
        images=data["images"],
        original_coords=data["original_coords"],
        base_image_path_list=list(data["base_image_path_list"]),
        extrinsic=data["extrinsic"],
        intrinsic=data["intrinsic"],
        depth_map=data["depth_map"],
        depth_conf=data["depth_conf"],
        points_3d=data["points_3d"],
        device=device,
        dtype=dtype,
        vggt_fixed_resolution=data["vggt_fixed_resolution"],
        img_load_resolution=data["img_load_resolution"],
        pred_tracks=data.get("pred_tracks"),
        pred_vis_scores=data.get("pred_vis_scores"),
        pred_confs=data.get("pred_confs"),
        tracked_points_3d=data.get("tracked_points_3d"),
        points_rgb=data.get("points_rgb"),
    )


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        use_autocast = images.device.type == "cuda"
        autocast_device_type = "cuda" if use_autocast else "cpu"
        autocast_dtype = dtype if use_autocast else None

        with torch.amp.autocast(
            autocast_device_type, dtype=autocast_dtype, enabled=use_autocast
        ):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


@dataclass
class VGGTLoopResult:
    images: torch.Tensor
    original_coords: torch.Tensor
    base_image_path_list: List[str]
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_map: np.ndarray
    depth_conf: np.ndarray
    points_3d: np.ndarray
    device: torch.device
    dtype: torch.dtype
    vggt_fixed_resolution: int
    img_load_resolution: int
    pred_tracks: Optional[np.ndarray] = None
    pred_vis_scores: Optional[np.ndarray] = None
    pred_confs: Optional[np.ndarray] = None
    tracked_points_3d: Optional[np.ndarray] = None
    points_rgb: Optional[np.ndarray] = None


def run_vggt_stage(args: argparse.Namespace) -> VGGTLoopResult:
    device, dtype = determine_device_and_dtype()
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("Model loaded")

    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    vggt_fixed_resolution = int(os.environ.get("VGGT_FIXED_RES", "518"))
    img_load_resolution = int(os.environ.get("VGGT_IMG_RES", "1024"))
    if img_load_resolution < vggt_fixed_resolution:
        img_load_resolution = vggt_fixed_resolution

    print(
        f"VGGT image resolution set to {img_load_resolution} (fixed={vggt_fixed_resolution})"
    )

    images, original_coords = load_and_preprocess_images_square(
        image_path_list, img_load_resolution
    )
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
        model, images, dtype, vggt_fixed_resolution
    )
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    should_predict_tracks = args.stage == "vggt" or args.use_ba
    skip_tracks_env = os.environ.get("VGGT_SKIP_TRACKS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if skip_tracks_env and should_predict_tracks:
        print("Skipping VGGT track prediction (VGGT_SKIP_TRACKS enabled)")
        should_predict_tracks = False
    pred_tracks = None
    pred_vis_scores = None
    pred_confs = None
    tracked_points_3d = None
    track_points_rgb = None

    if should_predict_tracks:
        print("Running track prediction as part of VGGT stage")
        images_for_tracking = images.to(device=device, dtype=dtype)

        use_autocast = device.type == "cuda"
        autocast_device_type = "cuda" if use_autocast else "cpu"
        autocast_dtype = dtype if use_autocast else None

        with torch.amp.autocast(
            autocast_device_type, dtype=autocast_dtype, enabled=use_autocast
        ):
            (
                pred_tracks,
                pred_vis_scores,
                pred_confs,
                tracked_points_3d,
                track_points_rgb,
            ) = predict_tracks(
                images_for_tracking,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return VGGTLoopResult(
        images=images,
        original_coords=original_coords,
        base_image_path_list=base_image_path_list,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_conf=depth_conf,
        points_3d=points_3d,
        device=device,
        dtype=dtype,
        vggt_fixed_resolution=vggt_fixed_resolution,
        img_load_resolution=img_load_resolution,
        pred_tracks=pred_tracks,
        pred_vis_scores=pred_vis_scores,
        pred_confs=pred_confs,
        tracked_points_3d=tracked_points_3d,
        points_rgb=track_points_rgb,
    )


def run_bundle_adjustment_stage(args: argparse.Namespace, vggt_result: VGGTLoopResult):
    image_size = np.array(vggt_result.images.shape[-2:])
    scale = vggt_result.img_load_resolution / vggt_result.vggt_fixed_resolution
    shared_camera = args.shared_camera

    if (
        vggt_result.pred_tracks is None
        or vggt_result.pred_vis_scores is None
        or vggt_result.tracked_points_3d is None
        or vggt_result.points_rgb is None
    ):
        raise ValueError(
            "Track predictions missing from VGGT stage. Run the VGGT stage with --use_ba or --stage vggt first."
        )

    pred_tracks = vggt_result.pred_tracks
    pred_vis_scores = vggt_result.pred_vis_scores
    tracked_points_3d = vggt_result.tracked_points_3d
    points_rgb = vggt_result.points_rgb

    intrinsic = vggt_result.intrinsic.copy()
    intrinsic[:, :2, :] *= scale
    track_mask = pred_vis_scores > args.vis_thresh

    if args.min_inlier_per_frame > 0:
        min_per_frame = args.min_inlier_per_frame
        # Guarantee at least min_per_frame observations per frame by selecting
        # the highest-visibility tracks even if they fall below the threshold.
        for frame_idx in range(track_mask.shape[0]):
            current = track_mask[frame_idx]
            needed = min_per_frame - int(current.sum())
            if needed <= 0:
                continue

            scores = pred_vis_scores[frame_idx]
            if np.all(np.isnan(scores)):
                continue
            order = np.argsort(np.nan_to_num(scores))
            if order.size == 0:
                continue
            top_indices = order[-min_per_frame:]
            track_mask[frame_idx, top_indices] = True

    inlier_counts = track_mask.sum(axis=0)
    if not np.any(inlier_counts >= 2):
        print("BA WARNING: no tracks observed in >=2 frames; using dense track mask")
        track_mask = ~np.isnan(pred_vis_scores)

    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        tracked_points_3d,
        vggt_result.extrinsic,
        intrinsic,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
        shared_camera=shared_camera,
        camera_type=args.camera_type,
        points_rgb=points_rgb,
        min_inlier_per_frame=args.min_inlier_per_frame,
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_use_gpu = vggt_result.device.type == "cuda"
    if hasattr(ba_options, "use_gpu"):
        ba_options.use_gpu = ba_use_gpu
    if ba_use_gpu and hasattr(ba_options, "gpu_index"):
        default_gpu = str(
            vggt_result.device.index if vggt_result.device.index is not None else 0
        )
        tracker_devices = getattr(args, "tracker_devices", None)
        ba_options.gpu_index = (
            tracker_devices if tracker_devices is not None else default_gpu
        )
    try:
        solver_options = ba_options.solver_options
    except TypeError:
        solver_options = None
    if ba_use_gpu and solver_options is not None and hasattr(solver_options, "use_gpu"):
        solver_options.use_gpu = True
    if ba_use_gpu:
        gpu_index_info = (
            ba_options.gpu_index if hasattr(ba_options, "gpu_index") else "default"
        )
        print(f"Bundle Adjustment will run on GPU device(s): {gpu_index_info}")
    else:
        print("Bundle Adjustment will run on CPU")
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    reconstruction_resolution = vggt_result.img_load_resolution
    return (
        reconstruction,
        tracked_points_3d,
        points_rgb,
        shared_camera,
        reconstruction_resolution,
    )


def run_feedforward_stage(args: argparse.Namespace, vggt_result: VGGTLoopResult):
    conf_thres_value = args.conf_thres_value
    max_points_for_colmap = 100000
    shared_camera = False
    camera_type = "PINHOLE"

    image_size = np.array(
        [vggt_result.vggt_fixed_resolution, vggt_result.vggt_fixed_resolution]
    )
    num_frames, height, width, _ = vggt_result.points_3d.shape

    points_rgb_tensor = F.interpolate(
        vggt_result.images,
        size=(vggt_result.vggt_fixed_resolution, vggt_result.vggt_fixed_resolution),
        mode="bilinear",
        align_corners=False,
    )
    points_rgb = (points_rgb_tensor.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = vggt_result.depth_conf >= conf_thres_value
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    points_3d = vggt_result.points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        vggt_result.extrinsic,
        vggt_result.intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction_resolution = vggt_result.vggt_fixed_resolution
    return (
        reconstruction,
        points_3d,
        points_rgb,
        shared_camera,
        reconstruction_resolution,
    )


def save_reconstruction_outputs(
    scene_dir: str,
    vggt_result: VGGTLoopResult,
    reconstruction,
    points_3d,
    points_rgb,
    shared_camera,
    reconstruction_resolution,
):
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        vggt_result.base_image_path_list,
        vggt_result.original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    trimesh.PointCloud(points_3d, colors=points_rgb).export(
        os.path.join(scene_dir, "sparse/points.ply")
    )


def demo_fn(args):
    print("Arguments:", vars(args))
    set_random_seeds(args.seed)
    stage_cache_path = get_stage_cache_path(args)

    if args.stage == "vggt":
        vggt_result = run_vggt_stage(args)
        save_vggt_stage_outputs(vggt_result, stage_cache_path)

        if args.use_ba:
            print(
                "VGGT stage complete. Skipping reconstruction export because --use_ba is set."
            )
            print(
                f"Run again with --stage ba to perform bundle adjustment using {stage_cache_path}."
            )
            return True

        (
            reconstruction,
            points_3d,
            points_rgb,
            shared_camera,
            reconstruction_resolution,
        ) = run_feedforward_stage(args, vggt_result)

        save_reconstruction_outputs(
            args.scene_dir,
            vggt_result,
            reconstruction,
            points_3d,
            points_rgb,
            shared_camera,
            reconstruction_resolution,
        )

        return True

    if args.stage == "ba":
        if not args.use_ba:
            print("Enabling bundle adjustment for BA stage execution.")
            args.use_ba = True
        vggt_result = load_vggt_stage_outputs(args, stage_cache_path)
    else:
        vggt_result = run_vggt_stage(args)

    if args.use_ba:
        (
            reconstruction,
            points_3d,
            points_rgb,
            shared_camera,
            reconstruction_resolution,
        ) = run_bundle_adjustment_stage(args, vggt_result)
    else:
        (
            reconstruction,
            points_3d,
            points_rgb,
            shared_camera,
            reconstruction_resolution,
        ) = run_feedforward_stage(args, vggt_result)

    save_reconstruction_outputs(
        args.scene_dir,
        vggt_result,
        reconstruction,
        points_3d,
        points_rgb,
        shared_camera,
        reconstruction_resolution,
    )

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
