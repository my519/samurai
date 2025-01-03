# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.compute_device = compute_device
        self.current_frame = 0
        
        # 读取第一帧获取视频尺寸信息
        first_frame = cv2.imread(img_paths[0])
        if first_frame is None:
            raise ValueError(f"无法读取图片: {img_paths[0]}")
        self.video_height, self.video_width = first_frame.shape[:2]
        
    def __len__(self):
        return len(self.img_paths)
        
    def read_frame(self):
        """读取下一帧"""
        if self.current_frame >= len(self.img_paths):
            return None
            
        try:
            img_path = self.img_paths[self.current_frame]
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"无法读取图片: {img_path}")
                return None
                
            # 转换为RGB并调整大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            
            # 转换为tensor并标准化
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame = (frame - self.img_mean) / self.img_std
            
            if not self.offload_video_to_cpu:
                frame = frame.to(self.compute_device)
            
            self.current_frame += 1
            return frame.unsqueeze(0)
            
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return None
            
    def reset(self):
        """重置到开始位置"""
        self.current_frame = 0


def load_video_frames(
    input_video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    output_video_path=None,
    start_Frame_idx=0,
):
    """
    Load the video frames from input_video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_bytes = isinstance(input_video_path, bytes)
    is_str = isinstance(input_video_path, str)
    is_mp4_path = is_str and os.path.splitext(input_video_path)[-1] in [".mp4", ".MP4"]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            input_video_path=input_video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
            output_video_path=output_video_path,
        )
    elif is_str and os.path.isdir(input_video_path):
        return load_video_frames_from_jpg_images(
            input_video_path=input_video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
            output_video_path=output_video_path,
            start_Frame_idx=start_Frame_idx,
        )
    else:
        raise NotImplementedError(
            "Only MP4 video and JPEG folder are supported at this moment"
        )


def load_video_frames_from_jpg_images(
    input_video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    output_video_path=None,
    start_Frame_idx=0,
):
    """使用流式方式加载JPG序列"""
    if not os.path.isdir(input_video_path):
        raise NotImplementedError(
            "输入路径必须是包含JPG序列的目录"
        )

    # 获取所有JPG文件
    frame_names = get_jpg_files(input_video_path, start_Frame_idx)
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"在 {input_video_path} 中没有找到JPG文件")
        
    img_paths = [os.path.join(input_video_path, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # 使用流式加载器
    lazy_images = AsyncVideoFrameLoader(
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    )
    return lazy_images, lazy_images.video_height, lazy_images.video_width


def get_compute_device(compute_device=None):
    if compute_device is not None:
        return compute_device
        
    if torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VideoStreamLoader:
    """流式视频加载器"""
    def __init__(self, input_video_path, image_size, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
        if isinstance(input_video_path, bytes):
            import numpy as np
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(input_video_path)
                temp_path = f.name
            self.cap = cv2.VideoCapture(temp_path)
            os.unlink(temp_path)
        else:
            self.cap = cv2.VideoCapture(input_video_path)
            self.input_video_path = input_video_path  # 保存路径以便重新打开
            
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_video_path}")
        
        # 获取视频属性时添加错误检查
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if width is None or height is None or frames is None:
            raise ValueError("无法获取视频属性")
            
        self.video_width = int(width)
        self.video_height = int(height)
        self.total_frames = int(frames)
        
        self.current_frame = 0
        
        if self.total_frames <= 0:
            raise ValueError("无法获取有效的视频帧数")
            
        self.image_size = image_size
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
    def __len__(self):
        return self.total_frames
    
    def read_frame(self):
        """读取下一帧"""
        try:
            if self.current_frame >= self.total_frames:
                return None
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            
            # 转换为RGB并调整大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            
            # 转换为tensor并标准化
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame = (frame - self.img_mean) / self.img_std
            
            self.current_frame += 1
            return frame.unsqueeze(0)
            
        except Exception as e:
            print(f"处理视频帧时出错: {str(e)}")
            return None
        
    def reset(self):
        """重置到视频开始"""
        #print(f"重置视频到开始位置 (当前帧={self.current_frame})")
        self.current_frame = 0
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not success:
            print("重置视频位置失败")
            # 尝试重新打开视频
            self.cap.release()
            self.cap = cv2.VideoCapture(self.input_video_path)
            if not self.cap.isOpened():
                print("重新打开视频失败")
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
            print("释放视频资源")

def load_video_frames_from_video_file(
    input_video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=None,
    output_video_path=None,
):
    """使用流式加载器加载视频"""
    try:
        loader = VideoStreamLoader(input_video_path, image_size, img_mean, img_std)
        return loader, loader.video_height, loader.video_width
    except Exception as e:
        print(f"加载视频时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    return mask
    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


def get_max_frame_number(output_fg_path,bprint = False):
    """获取目录中jpg文件名中最大的数字
    
    Args:
        output_video_path: 输出目录路径
        
    Returns:
        max_num: 最大的文件序号，如果目录不存在或没有jpg文件则返回-1
    """
    try:
        if not os.path.isdir(output_fg_path):
            return 0
            
        # 获取所有JPG文件
        frame_names = [
            p
            for p in os.listdir(output_fg_path)
            if p.lower().endswith(('.jpg', '.jpeg'))
        ]
        
        if not frame_names:
            print(f"目录 {output_fg_path} 中没有找到JPG文件")
            return 0
            
        # 从文件名中提取数字并找到最大值
        max_num = 0
        frame_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        max_num = int(''.join(filter(str.isdigit, frame_names[-1]))) if frame_names else -1
        if bprint:
            print(f"已处理JPG文件最大帧号: {max_num}")
        return max_num
        
    except Exception as e:
        print(f"获取最大帧号时出错: {str(e)}")
        return 0

def get_all_frame_box_number(video_input_path,bprint = False):
    """获取所有帧的box序号
    
    Args:
        video_input_path: 输入目录路径
        
    Returns:
        numeric_frame_names: 所有帧的box序号，如果目录不存在或没有txt文件则返回空列表
    """
    try:
        if not os.path.isdir(video_input_path):
            print(f"目录 {video_input_path} 不存在")
            return []
            
        # 获取所有JPG文件
        frame_names = [
            p
            for p in os.listdir(video_input_path)
            if p.lower().endswith(('.txt', '.txt'))
        ]
        
        if not frame_names:
            print(f"目录 {video_input_path} 中没有找到目标框txt文件")
            return []
            
        # 从文件名中提取数字并找到最大值
        frame_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        numeric_frame_names = [int(''.join(filter(str.isdigit, x))) for x in frame_names]
        if bprint:
            print(f"找到的所有目标框帧号: {numeric_frame_names}")
        return numeric_frame_names
        
    except Exception as e:
        print(f"获取所有帧号时出错: {str(e)}")
        return []


def get_start_frame_number(video_input_path,output_fg_path,bprint = False):
    """获取开始帧号
    
    Args:
        video_input_path: 输入目录路径
        output_video_path: 输出目录路径
        
    Returns:
        start_frame_number: 开始帧号，如果目录不存在或没有txt文件则返回-1
    """
    numeric_frame_names = get_all_frame_box_number(video_input_path,bprint)
    max_num = get_max_frame_number(output_fg_path,bprint)
    if numeric_frame_names and max_num:
        numeric_frame_names.sort(reverse=True)
        for frame_number in numeric_frame_names:
            if frame_number < max_num:
                if bprint:
                    print(f"从第{frame_number}帧开始处理")
                return frame_number
        return 0
    return 0

def get_jpg_files_bak(video_input_path, output_fg_path, bprint=False):
    if output_fg_path is None or video_input_path is None:
        return []
        
    # 获取JPG序列信息
    jpg_files = [f for f in os.listdir(video_input_path) 
                if f.lower().endswith(('.jpg', '.jpeg'))]
    if not jpg_files:
        raise ValueError(f"目录 {video_input_path} 中没有找到JPG文件")
    
    # 使用自然排序（数值排序），处理没有数字的情况
    jpg_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    start_frame_number = get_start_frame_number(video_input_path, output_fg_path, bprint)
    
    if start_frame_number > 0:
        # 找到起始帧文件名
        jpg_files = jpg_files[jpg_files.index(f"{start_frame_number}.jpg"):]
            
    return jpg_files

def get_jpg_files(video_input_path, start_Frame_idx, bprint=False):
    if video_input_path is None:
        return []
        
    # 获取JPG序列信息
    jpg_files = [f for f in os.listdir(video_input_path) 
                if f.lower().endswith(('.jpg', '.jpeg'))]
    if not jpg_files:
        raise ValueError(f"目录 {video_input_path} 中没有找到JPG文件")
    
    # 使用自然排序（数值排序），处理没有数字的情况
    jpg_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    if start_Frame_idx > 0:
        # 找到起始帧文件名
        jpg_files = jpg_files[jpg_files.index(f"{start_Frame_idx}.jpg"):]
            
    return jpg_files

