import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm
import traceback

color = [(255, 0, 0)]

def load_txt(gt_path):
    prompts = {}
    try:
        with open(gt_path, 'r') as f:
            gt = f.readlines()
        for fid, line in enumerate(gt):
            x, y, w, h = map(float, line.split(','))
            x, y, w, h = int(x), int(y), int(w), int(h)
            prompts[fid] = ((x, y, x + w, y + h), 0)
    except Exception as e:
        pass
    return prompts

def load_frame_box_video(video_path, frame_idx):
    video_path = os.path.dirname(video_path)
    txt_file_path = os.path.join(video_path, f"{frame_idx}.txt")
    box = load_txt(txt_file_path)
    if box:
        print(f"加载帧{frame_idx}的边界框: {txt_file_path}:{box}")
    return box

def load_frame_box_jpeg(video_path, jpg_file):
    video_path = os.path.dirname(video_path)
    jpg_filename = os.path.splitext(jpg_file)[0]
    txt_file_path = os.path.join(video_path, f"{jpg_filename}.txt")
    box = load_txt(txt_file_path)
    if box:
        print(f"加载帧{jpg_file}的边界框: {txt_file_path}:{box}")
    return box

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def get_device(force_cpu=False):
    """检测并返回可用的设备
    
    Args:
        force_cpu: 如果为True，强制使用CPU，忽略GPU
    """
    if force_cpu:
        print("强制使用CPU")
        return torch.device("cpu")
        
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            print("使用Intel GPU")
            return torch.device("xpu")
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        print("使用NVIDIA GPU")
        return torch.device("cuda")
    
    print("使用CPU")
    return torch.device("cpu")

def main(args):
    try:
        # 初始化模型
        model_cfg = determine_model_cfg(args.model_path)
        device = get_device(args.force_cpu)
        print(f"使用设备: {device}")
        
        # 初始化预测器
        if device.type == "xpu":
            import intel_extension_for_pytorch as ipex
            predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device)
            predictor = ipex.optimize(predictor)
        else:
            predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device)
        
        # 检查输入路径是视频还是图片目录
        is_video = os.path.isfile(args.video_path) and args.video_path.lower().endswith(('.mp4', '.MP4'))
        is_jpg_dir = os.path.isdir(args.video_path)
        
        if is_video:
            # 获取视频信息
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frames is None:
                raise ValueError("无法获取视频总帧数")
                
            total_frames = int(frames)
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("无法读取视频")
            height, width = first_frame.shape[:2]
            cap.release()
        elif is_jpg_dir:
            # 获取JPG序列信息
            jpg_files = [f for f in os.listdir(args.video_path) 
                        if f.lower().endswith(('.jpg', '.jpeg'))]
            if not jpg_files:
                raise ValueError(f"目录 {args.video_path} 中没有找到JPG文件")
            
            # 使用自然排序（数值排序），处理没有数字的情况
            jpg_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
            total_frames = len(jpg_files)
            print(f"找到 {total_frames} 个JPG文件")

            # 读取第一张图片获取尺寸信息
            first_frame = cv2.imread(os.path.join(args.video_path, jpg_files[0]))
            if first_frame is None:
                raise ValueError(f"无法读取图片: {jpg_files[0]}")
            height, width = first_frame.shape[:2]
            frame_rate = 30  # 默认帧率
        else:
            raise ValueError("输入路径必须是MP4视频文件或包含JPG序列的目录")
        
        # 加载提示信息
        #prompts = load_txt(args.txt_path)
        prompts=load_frame_box_video(args.video_path,0)

        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

        # 创建显示窗口
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', int(width/2), int(height/2))
        cv2.resizeWindow('Mask', int(width/2), int(height/2))
        cv2.resizeWindow('Result', int(width/2), int(height/2))

        with torch.inference_mode():
            print("初始化状态...")
            state = predictor.init_state(
                args.video_path,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True
            )

            print("添加第一帧的边界框...")
            bbox, track_label = prompts[0]
            print(f"第一帧边界框: {bbox}")
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            print("开始处理视频帧...")
            if is_video:
                cap = cv2.VideoCapture(args.video_path)
            else:
                current_frame_idx = 0
                
            for frame_idx, object_ids, masks in tqdm(
                predictor.propagate_in_video(state),
                total=total_frames,
                desc="Processing frames"
            ):
                if is_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    # 读取JPG序列中的当前帧
                    frame_path = os.path.join(args.video_path, jpg_files[current_frame_idx])
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"无法读取图片: {frame_path}")
                        break
                    current_frame_idx += 1

                # 尝试读入与帧序号frame_idx同名的txt文件，作为提示信息
                if frame_idx > 0:
                    if is_video:
                        prompts = load_frame_box_video(args.video_path, frame_idx)
                    else:
                        prompts = load_frame_box_jpeg(args.video_path, jpg_files[current_frame_idx])

                    if prompts:
                        try:
                            print(f"重置第 {frame_idx} 帧的跟踪状态")
                            
                            # 保存当前设备信息
                            current_device = device
                            
                            # 重置状态前先清理资源
                            if 'state' in locals():
                                del state
                            torch.cuda.empty_cache()  # 清理GPU内存
                            gc.collect()  # 清理CPU内存
                            
                            # 重新初始化状态
                            state = predictor.init_state(
                                args.video_path,
                                offload_video_to_cpu=True,
                                offload_state_to_cpu=True
                            )
                            
                            # 添加新的边界框
                            bbox, track_label = prompts[0]
                            print(f"添加新的边界框: {bbox}")
                            
                            # 检查边界框的有效性
                            x1, y1, x2, y2 = bbox
                            if (x1 >= 0 and y1 >= 0 and x2 < width and y2 < height and 
                                x2 > x1 and y2 > y1):
                                # 添加新的边界框并获取掩码
                                _, _, new_masks = predictor.add_new_points_or_box(
                                    state, 
                                    box=bbox, 
                                    frame_idx=frame_idx, 
                                    obj_id=0
                                )
                                
                                # 检查掩码的有效性
                                if new_masks is not None and len(new_masks) > 0:
                                    print(f"更新第 {frame_idx} 帧的掩码")
                                    masks = new_masks  # 更新当前掩码
                                else:
                                    print(f"警告: 第 {frame_idx} 帧未能生成有效掩码")
                                    continue
                            else:
                                print(f"警告: 无效的边界框坐标: {bbox}")
                                continue
                                
                        except Exception as e:
                            print(f"重置跟踪状态时出错: {str(e)}")
                            traceback.print_exc()  # 打印完整的错误堆栈
                            continue

                # 处理每个对象的mask
                for obj_id, mask in zip(object_ids, masks):
                    try:
                        # 转换mask为numpy数组
                        mask = mask[0].cpu().numpy()
                        mask = mask > 0.0
                        
                        # 创建结果图像（默认使用原始图像）
                        result = frame.copy()

                        # 处理按键
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):  # 按q退出
                            raise StopIteration
                        elif key == ord(' '):  # 按空格暂停
                            while True:
                                key = cv2.waitKey(0) & 0xFF
                                if key == ord(' '):  # 再次按空格继续
                                    break
                                elif key == ord('q'):  # 按q退出
                                    raise StopIteration                                  
                        
                        # 检查mask是否为空
                        if not np.any(mask):
                            # 添加帧信息
                            info_text = f'Frame: {frame_idx} (No Mask)'
                            cv2.putText(result, info_text, (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # 如果有ground truth边界框，仍然显示它
                            if frame_idx in prompts:
                                x1, y1, x2, y2 = prompts[frame_idx][0]
                                cv2.rectangle(result, (x1, y1), (x2, y2), 
                                            (0, 255, 0), 2)  # 绿色表示真实框
                            
                            # 显示和保存结果
                            cv2.imshow('Original', frame)
                            cv2.imshow('Mask', np.zeros_like(frame))  # 显示空mask
                            cv2.imshow('Result', result)
                            
                            if args.save_to_video:
                                out.write(result)
                            continue
                        
                        # 创建mask可视化
                        mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
                        mask_vis[mask] = (0, 255, 0)  # 绿色表示mask域

                        # 检查数组形状和非空性
                        if (result[mask].size > 0 and mask_vis[mask].size > 0 and 
                            result[mask].shape == mask_vis[mask].shape):
                            # 叠加mask
                            result[mask] = cv2.addWeighted(result[mask], 0.7, mask_vis[mask], 0.3, 0)
                        else:
                            # 添加帧信息
                            info_text = f'Frame: {frame_idx} (Invalid Mask)'
                            cv2.putText(result, info_text, (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # 如果有ground truth边界框，仍然显示它
                            if frame_idx in prompts:
                                x1, y1, x2, y2 = prompts[frame_idx][0]
                                cv2.rectangle(result, (x1, y1), (x2, y2), 
                                            (0, 255, 0), 2)  # 绿色表示真实框
                            
                            # 显示和保存结果
                            cv2.imshow('Original', frame)
                            cv2.imshow('Mask', np.zeros_like(frame))
                            cv2.imshow('Result', result)
                            
                            if args.save_to_video:
                                out.write(result)
                            continue

                        # 取边界框
                        non_zero_indices = np.argwhere(mask)
                        if len(non_zero_indices) > 0:
                            y_min, x_min = non_zero_indices.min(axis=0)
                            y_max, x_max = non_zero_indices.max(axis=0)
                            # 绘制预测的边界框
                            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), 
                                        (0, 0, 255), 2)  # 红色表示预测框

                        # 绘制ground truth边界框
                        if frame_idx in prompts:
                            x1, y1, x2, y2 = prompts[frame_idx][0]
                            cv2.rectangle(result, (x1, y1), (x2, y2), 
                                        (0, 255, 0), 2)  # 绿色表示真实框

                        # 添加帧信息
                        info_text = f'Frame: {frame_idx}'
                        cv2.putText(result, info_text, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # 显示图像
                        cv2.imshow('Original', frame)
                        cv2.imshow('Mask', mask_vis)
                        cv2.imshow('Result', result)

                        # 保存结果
                        if args.save_to_video:
                            out.write(result)

                    except Exception as e:
                        print(f"处理第 {frame_idx} 帧时发生错误: {str(e)}")
                        continue

    except StopIteration:
        print("\n用户终止处理")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        
        if 'predictor' in locals():
            del predictor
        if 'state' in locals():
            del state
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--video_path", default="D:/2024Work/14.AIVideo/6.Assets/6.Video/hero.jpg/", help="Input video path.")
    parser.add_argument("--video_path", required=True, help="Input video path.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--video_output_path", default="demo.mp4")
    parser.add_argument("--save_to_video", default=True, type=bool)
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU运行，忽略GPU")
    args = parser.parse_args()
    main(args)
