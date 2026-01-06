#!/usr/bin/env python3
"""
从视差数据生成点云脚本

使用方法:
    python disp_to_pointcloud.py \\
        --left A_xxx.jpg \\
        --right D_xxx.jpg \\
        --calib K.txt \\
        --disp disp.npz \\
        --output output_cloud.ply

说明:
    - 读取左右图像、相机标定文件和视差数据
    - 计算深度图
    - 生成彩色点云（PLY格式）
    - 支持 .npy 和 .npz 格式的视差文件
"""

import argparse
import os
import numpy as np
import cv2
from pathlib import Path


def parse_calib(calib_path):
    """解析相机标定文件 (支持多种 .txt 和 .xml 格式)"""
    calib_w = None
    calib_h = None

    if calib_path.endswith('.txt'):
        with open(calib_path, 'r') as f:
            content = f.read().strip()
        
        # 检测格式类型
        if '=' in content:
            # Middlebury 格式: cam0=[...] baseline=...
            calib_data = {}
            for line in content.split('\n'):
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key in ['cam0', 'cam1']:
                        # 解析内参矩阵
                        value = value.strip('[]')
                        rows = value.split(';')
                        matrix = []
                        for row in rows:
                            matrix.append([float(x) for x in row.split()])
                        calib_data[key] = np.array(matrix)
                    elif key in ['doffs', 'baseline', 'width', 'height', 'ndisp']:
                        calib_data[key] = float(value)
            
            # 提取关键参数
            fx = calib_data['cam0'][0, 0]
            fy = calib_data['cam0'][1, 1]
            cx = calib_data['cam0'][0, 2]
            cy = calib_data['cam0'][1, 2]
            baseline = calib_data['baseline']  # 单位：毫米
            doffs = calib_data.get('doffs', 0.0)  # 视差偏移
            # Optional size info (pixels)
            if 'width' in calib_data and 'height' in calib_data:
                calib_w = int(calib_data['width'])
                calib_h = int(calib_data['height'])
            
        else:
            # 简单格式: 第一行是 3x3 内参矩阵（按行展开），第二行是 baseline
            # 格式: fx 0 cx 0 fy cy 0 0 1
            #      baseline
            lines = content.split('\n')
            if len(lines) < 2:
                raise ValueError("标定文件格式错误：需要至少两行（内参矩阵 + baseline）")
            
            # 解析内参矩阵
            intrinsics = [float(x) for x in lines[0].split()]
            if len(intrinsics) != 9:
                raise ValueError(f"内参矩阵应包含9个数字，实际有 {len(intrinsics)} 个")
            
            fx = intrinsics[0]
            cx = intrinsics[2]
            fy = intrinsics[4]
            cy = intrinsics[5]
            
            # 解析 baseline（单位：米，需转换为毫米）
            baseline = float(lines[1].strip()) * 1000.0  # 米 -> 毫米
            doffs = 0.0  # 简单格式默认 doffs=0
        
        return fx, fy, cx, cy, baseline, doffs, calib_w, calib_h
    
    elif calib_path.endswith('.xml'):
        # XML 格式
        import xml.etree.ElementTree as ET
        tree = ET.parse(calib_path)
        root = tree.getroot()
        
        fx = float(root.find('.//fx').text)
        fy = float(root.find('.//fy').text)
        cx = float(root.find('.//cx').text)
        cy = float(root.find('.//cy').text)
        baseline = float(root.find('.//baseline').text)
        doffs_node = root.find('.//doffs')
        doffs = float(doffs_node.text) if doffs_node is not None else 0.0
        
        return fx, fy, cx, cy, baseline, doffs, calib_w, calib_h
    
    else:
        raise ValueError(f"不支持的标定文件格式: {calib_path}")


def load_disparity(disp_path):
    """加载视差数据
    
    支持格式:
        - .npy: numpy array
        - .npz: 压缩numpy，自动查找 'disparity', 'disp', 'arr_0' 等键
    """
    if disp_path.endswith('.npy'):
        disparity = np.load(disp_path)
        print(f"✓ 加载视差数据: {disp_path}")
        print(f"  形状: {disparity.shape}, dtype: {disparity.dtype}")
        return disparity
    
    elif disp_path.endswith('.npz'):
        data = np.load(disp_path)
        # 尝试常见的键名
        possible_keys = ['disparity', 'disp', 'arr_0', 'disparity_raw', 'disp_raw']
        for key in possible_keys:
            if key in data:
                disparity = data[key]
                print(f"✓ 加载视差数据: {disp_path} (键: '{key}')")
                print(f"  形状: {disparity.shape}, dtype: {disparity.dtype}")
                return disparity
        
        # 如果没找到，列出所有可用的键
        available_keys = list(data.keys())
        raise ValueError(f"无法在 .npz 文件中找到视差数据。可用的键: {available_keys}")
    
    else:
        raise ValueError(f"不支持的视差文件格式: {disp_path}")


def compute_depth(disparity, fx, baseline, doffs=0.0, 
                 conf=None, occ=None, 
                 conf_thresh=0.0, occ_thresh=0.0,
                 depth_trunc_m=20.0, min_disp=1e-6):
    """从视差计算深度
    
    Args:
        disparity: 视差图 (H, W)
        fx: 焦距
        baseline: 基线距离（毫米）
        doffs: 视差偏移
        conf: 置信度图 (可选)
        occ: 遮挡图 (可选)
        conf_thresh: 置信度阈值
        occ_thresh: 遮挡阈值
        depth_trunc_m: 深度截断（米）
        min_disp: 最小视差（防止除零）
    
    Returns:
        depth: 深度图 (H, W)，单位：毫米
        valid_mask: 有效像素掩码
    """
    # 计算有效掩码
    total_pixels = disparity.size
    if conf is not None and occ is not None:
        valid_mask = (conf > conf_thresh) & (occ > occ_thresh) & (disparity > 0)
        print(f"  过滤统计:")
        print(f"    总像素: {total_pixels:,}")
        print(f"    置信度>{conf_thresh}: {(conf > conf_thresh).sum():,} ({(conf > conf_thresh).sum()/total_pixels*100:.1f}%)")
        print(f"    遮挡>{occ_thresh}: {(occ > occ_thresh).sum():,} ({(occ > occ_thresh).sum()/total_pixels*100:.1f}%)")
        print(f"    视差>0: {(disparity > 0).sum():,} ({(disparity > 0).sum()/total_pixels*100:.1f}%)")
        print(f"    初始有效像素: {valid_mask.sum():,} ({valid_mask.sum()/total_pixels*100:.1f}%)")
    else:
        valid_mask = disparity > 0
        print(f"  视差>0 的像素: {valid_mask.sum():,} ({valid_mask.sum()/total_pixels*100:.1f}%)")
    
    # 计算深度: depth_mm = baseline_mm * fx / (disp + doffs)
    disp32 = disparity.astype(np.float32, copy=False)
    fx_f = float(fx)
    baseline_f = float(baseline)
    doffs_f = float(doffs)

    denom = disp32 + doffs_f
    depth = np.zeros_like(disp32, dtype=np.float32)

    # 仅在 valid_mask 内计算，并防止 denom 过小
    denom_valid = denom[valid_mask]
    denom_valid = np.maximum(denom_valid, float(min_disp))
    depth[valid_mask] = (baseline_f * fx_f) / denom_valid

    # 深度截断
    depth_trunc = float(depth_trunc_m) * 1000.0  # 米 -> 毫米
    invalid_depth_mask = (depth > depth_trunc) | (depth <= 0) | (~np.isfinite(depth))
    filtered_by_trunc = int((valid_mask & (depth > depth_trunc)).sum())
    filtered_by_nonpos = int((valid_mask & (depth <= 0)).sum())
    filtered_by_naninf = int((valid_mask & (~np.isfinite(depth))).sum())
    depth[invalid_depth_mask] = 0

    print(f"    深度截断(>{depth_trunc_m:g}m): {filtered_by_trunc:,} 像素被过滤")
    if filtered_by_nonpos or filtered_by_naninf:
        print(f"    非法深度(<=0): {filtered_by_nonpos:,}，NaN/Inf: {filtered_by_naninf:,}")
    
    # 更新有效掩码
    valid_mask = valid_mask & ~invalid_depth_mask
    print(f"    最终有效像素: {valid_mask.sum():,} ({valid_mask.sum()/total_pixels*100:.1f}%)")
    
    if valid_mask.any():
        print(f"  深度范围: [{depth[valid_mask].min()/1000.0:.3f}, {depth[valid_mask].max()/1000.0:.3f}] 米")
    
    return depth, valid_mask


def depth_to_pointcloud(depth, left_image, fx, fy, cx, cy, depth_trunc_m=20.0):
    """从深度图生成彩色点云
    
    Args:
        depth: 深度图 (H, W)，单位：毫米
        left_image: 左视图 RGB 图像 (H, W, 3)
        fx, fy: 焦距
        cx, cy: 主点坐标
        depth_trunc_m: 深度截断（米）
    
    Returns:
        points: (N, 3) 3D 坐标（单位：米）
        colors: (N, 3) RGB 颜色 (0-255)
    """
    if depth is None or left_image is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    try:
        h, w = depth.shape
        if h == 0 or w == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        # 优先使用 Open3D（速度更快）
        try:
            import open3d as o3d
            use_open3d = True
        except ImportError:
            use_open3d = False
            print("  提示: 未安装 Open3D，使用 numpy fallback（较慢）")

        if use_open3d:
            depth_o3d = o3d.geometry.Image(depth.astype(np.float32, copy=False))
            rgb_o3d = o3d.geometry.Image(left_image.astype(np.uint8, copy=False))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=1000.0,
                depth_trunc=float(depth_trunc_m),
                convert_rgb_to_intensity=False,
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                int(w), int(h), 
                float(fx), float(fy), 
                float(cx), float(cy)
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = (np.asarray(pcd.colors, dtype=np.float32) * 255.0)
            colors = np.clip(colors, 0, 255).astype(np.uint8)
            return points, colors

        # Fallback (无 Open3D)
        valid_mask = depth > 0
        if not np.any(valid_mask):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u[valid_mask].astype(np.float32)
        v = v[valid_mask].astype(np.float32)
        Z = (depth[valid_mask].astype(np.float32) / 1000.0)  # mm -> m
        X = (u - float(cx)) * Z / float(fx)
        Y = (v - float(cy)) * Z / float(fy)
        points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        colors = left_image[valid_mask].astype(np.uint8)
        return points, colors
    
    except Exception as e:
        print(f"  ⚠ 警告: 点云生成失败: {e}")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)


def save_pointcloud(points, colors, output_path):
    """保存点云为 PLY 格式
    
    Args:
        points: (N, 3) 3D 坐标（单位：米）
        colors: (N, 3) RGB 颜色 (0-255)
        output_path: 输出路径
    """
    n_points = points.shape[0]
    
    # 保存 ASCII 格式
    with open(output_path, 'w') as f:
        # 写入头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入数据
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"✓ 保存点云: {output_path}")
    print(f"  点数: {n_points:,}")


def _scale_intrinsics_to_resolution(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    """缩放内参到目标分辨率"""
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return fx, fy, cx, cy
    if src_w == dst_w and src_h == dst_h:
        return fx, fy, cx, cy
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def main():
    parser = argparse.ArgumentParser(
        description='从视差数据生成点云',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 基本使用
    python disp_to_pointcloud.py \\
        --left A_xxx.jpg \\
        --right D_xxx.jpg \\
        --calib K.txt \\
        --disp disp.npy

    # 指定输出路径
    python disp_to_pointcloud.py \\
        --left A_xxx.jpg \\
        --calib K.txt \\
        --disp disp.npz \\
        --output my_cloud.ply

    # 使用置信度和遮挡过滤
    python disp_to_pointcloud.py \\
        --left A_xxx.jpg \\
        --calib K.txt \\
        --disp disp.npy \\
        --conf conf.npy \\
        --occ occ.npy \\
        --conf_thresh 0.3 \\
        --occ_thresh 0.1

    # 调整深度截断
    python disp_to_pointcloud.py \\
        --left A_xxx.jpg \\
        --calib K.txt \\
        --disp disp.npy \\
        --depth_trunc_m 5.0
        '''
    )
    
    # 必需参数
    parser.add_argument('--left', required=True, help='左图像路径 (A图像)')
    parser.add_argument('--calib', required=True, help='相机标定文件路径 (K.txt)')
    parser.add_argument('--disp', required=True, help='视差文件路径 (.npy 或 .npz)')
    
    # 可选参数
    parser.add_argument('--right', help='右图像路径 (D图像)，可选，仅用于验证')
    parser.add_argument('--output', help='输出点云路径 (默认: 自动生成)')
    parser.add_argument('--conf', help='置信度文件路径 (.npy/.npz)')
    parser.add_argument('--occ', help='遮挡文件路径 (.npy/.npz)')
    
    # 过滤参数
    parser.add_argument('--conf_thresh', type=float, default=0.0,
                       help='置信度阈值 (默认: 0.0，即不过滤)')
    parser.add_argument('--occ_thresh', type=float, default=0.0,
                       help='遮挡阈值 (默认: 0.0，即不过滤)')
    parser.add_argument('--depth_trunc_m', type=float, default=20.0,
                       help='深度截断（米），默认: 20.0')
    
    # 内参缩放参数
    parser.add_argument('--calib_width', type=int, help='标定文件对应的图像宽度（用于内参缩放）')
    parser.add_argument('--calib_height', type=int, help='标定文件对应的图像高度（用于内参缩放）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("从视差数据生成点云")
    print("="*70)
    
    # 1. 加载左图像
    print("\\n[1/5] 加载左图像...")
    if not os.path.exists(args.left):
        raise FileNotFoundError(f"左图像不存在: {args.left}")
    
    left_bgr = cv2.imread(args.left)
    if left_bgr is None:
        raise ValueError(f"无法读取左图像: {args.left}")
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = left_rgb.shape[:2]
    print(f"✓ 左图像: {args.left}")
    print(f"  分辨率: {img_w}x{img_h}")
    
    # 2. 加载视差数据
    print("\\n[2/5] 加载视差数据...")
    if not os.path.exists(args.disp):
        raise FileNotFoundError(f"视差文件不存在: {args.disp}")
    disparity = load_disparity(args.disp)
    
    # 检查形状匹配
    if disparity.shape != (img_h, img_w):
        print(f"  ⚠ 警告: 视差图尺寸 {disparity.shape} 与图像尺寸 ({img_h}, {img_w}) 不匹配")
        if disparity.size == img_h * img_w:
            print(f"  尝试 reshape...")
            disparity = disparity.reshape(img_h, img_w)
        else:
            raise ValueError(f"视差图尺寸不匹配且无法 reshape")
    
    print(f"  视差范围: [{disparity.min():.2f}, {disparity.max():.2f}] px")
    
    # 3. 加载置信度和遮挡（可选）
    conf = None
    occ = None
    
    if args.conf:
        print("\\n[3/5] 加载置信度数据...")
        if not os.path.exists(args.conf):
            print(f"  ⚠ 警告: 置信度文件不存在: {args.conf}")
        else:
            conf = load_disparity(args.conf)  # 复用加载函数
            if conf.shape != (img_h, img_w):
                conf = conf.reshape(img_h, img_w)
            print(f"  置信度范围: [{conf.min():.3f}, {conf.max():.3f}]")
    
    if args.occ:
        if not args.conf:
            print("\\n[3/5] 加载遮挡数据...")
        if not os.path.exists(args.occ):
            print(f"  ⚠ 警告: 遮挡文件不存在: {args.occ}")
        else:
            occ = load_disparity(args.occ)
            if occ.shape != (img_h, img_w):
                occ = occ.reshape(img_h, img_w)
            print(f"  遮挡范围: [{occ.min():.3f}, {occ.max():.3f}]")
    
    if not args.conf and not args.occ:
        print("\\n[3/5] 未提供置信度/遮挡数据，将使用所有视差>0的点")
    
    # 4. 解析标定文件
    print("\\n[4/5] 解析相机标定...")
    if not os.path.exists(args.calib):
        raise FileNotFoundError(f"标定文件不存在: {args.calib}")
    
    fx, fy, cx, cy, baseline, doffs, calib_w, calib_h = parse_calib(args.calib)
    
    # 内参缩放（如果图像分辨率与标定不同）
    src_w = args.calib_width if args.calib_width else (calib_w if calib_w else img_w)
    src_h = args.calib_height if args.calib_height else (calib_h if calib_h else img_h)
    
    if (src_w, src_h) != (img_w, img_h):
        fx, fy, cx, cy = _scale_intrinsics_to_resolution(
            fx, fy, cx, cy,
            src_w=src_w, src_h=src_h,
            dst_w=img_w, dst_h=img_h,
        )
        print(f"  内参缩放: ({src_w}x{src_h}) -> ({img_w}x{img_h})")
    
    print(f"✓ 标定参数:")
    print(f"  焦距: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  主点: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  基线: {baseline:.2f} mm")
    print(f"  视差偏移: {doffs:.2f}")
    
    # 5. 计算深度并生成点云
    print("\\n[5/5] 计算深度并生成点云...")
    
    # 计算深度
    depth, valid_mask = compute_depth(
        disparity, fx, baseline, doffs,
        conf=conf, occ=occ,
        conf_thresh=args.conf_thresh,
        occ_thresh=args.occ_thresh,
        depth_trunc_m=args.depth_trunc_m
    )
    
    if not valid_mask.any():
        print("\\n❌ 错误: 没有有效的深度点！")
        print("  建议:")
        print("  - 检查视差数据是否正确")
        print("  - 降低 --conf_thresh 和 --occ_thresh")
        print("  - 增大 --depth_trunc_m")
        return
    
    # 生成点云
    print("\\n  生成3D点云...")
    points, colors = depth_to_pointcloud(
        depth, left_rgb, fx, fy, cx, cy,
        depth_trunc_m=args.depth_trunc_m
    )
    
    if points.shape[0] == 0:
        print("\\n❌ 错误: 点云生成失败（0个点）")
        return
    
    # 保存点云
    if args.output:
        output_path = args.output
    else:
        # 自动生成输出路径
        disp_stem = Path(args.disp).stem
        output_path = str(Path(args.disp).parent / f"{disp_stem}_cloud.ply")
    
    print("\\n  保存点云文件...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_pointcloud(points, colors, output_path)
    
    # 同时保存深度图 (.npy)
    depth_output = str(Path(output_path).with_suffix('')) + '_depth_meter.npy'
    np.save(depth_output, depth / 1000.0)
    print(f"✓ 保存深度数据: {depth_output}")
    
    print("\\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    print(f"\\n输出文件:")
    print(f"  点云: {output_path}")
    print(f"  深度: {depth_output}")
    print(f"\\n点云统计:")
    print(f"  点数: {points.shape[0]:,}")
    if points.shape[0] > 0:
        print(f"  X范围: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
        print(f"  Y范围: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
        print(f"  Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n⚠ 用户中断")
    except Exception as e:
        print(f"\\n\\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
