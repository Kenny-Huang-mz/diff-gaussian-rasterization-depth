#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians( # 一个python的接口，最终调用的还是_RasterizeGaussians.apply，一个PyTorch的自定义求导函数
    means3D, # 每个高斯椭球在 3D 世界坐标系中的中心点位置
    means2D, # 3D 中心点投影到 2D 图像平面上的坐标
    sh,
    colors_precomp, # 预计算的颜色，通常是球谐函数的零阶（DC）分量，代表基础颜色
    features_precomp,
    opacities,
    scales,
    rotations, # 每个高斯椭球的旋转姿态，通常用四元数表示
    cov3Ds_precomp, # 预计算的 3D 协方差矩阵，它结合了缩放和旋转，精确定义了高斯椭球在 3D 空间中的形状和方向
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        features_precomp, # 新增
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward( # 前向传播函数，计算渲染结果，将3D高斯绘制或渲染成2D图像，并保存必要的中间变量以供后向传播使用
        ctx,
        means3D,
        means2D,
        sh, # 球谐系数
        colors_precomp, # 预计算的颜色，即基于当前相机视角，利用sh系数计算出的颜色
        features_precomp, # 新增
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings, # 光栅化设置，包含与渲染过程相关的相机参数和控制参数，如相机位置、朝向、输出图像尺寸、视场切线值
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            features_precomp, # 新增
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix, # 相机视图矩阵，描述相机的朝向和位置
            raster_settings.projmatrix, # 相机投影矩阵，描述相机的投影方式
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos, #相机中心的位置，虽然在viewmatrix中也有，但这里是为了方便使用
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth_map, weight_map, feature_map = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth_map, weight_map, feature_map = _C.rasterize_gaussians(*args)
            #return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, features_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer,
                              depth_map, weight_map, feature_map)
        return color, radii, depth_map, weight_map, feature_map

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_depth, grad_weight, grad_feature_map):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, features_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, depth_map, weight_map, feature_map = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii,
                weight_map, # 在原3dgs基础上额外添加
                depth_map, # 在原3dgs基础上额外添加
                colors_precomp, 
                features_precomp,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_depth,  # 在原3dgs基础上额外添加
                grad_feature_map,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_features = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_features = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_features, # 新增
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, features_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings
        # 对于颜色，要么提供球谐函数系数 shs（让程序自己算颜色），要么提供预计算好的颜色 colors_precomp，但不能两个都给，也不能都不给。
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        # 对于几何形状，要么提供缩放 scales 和旋转 rotations（让程序自己算协方差），要么提供预计算好的 3D 协方差 cov3D_precomp，同样，不能两个都给或都不给。
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if features_precomp is None: # 新增
            features_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            features_precomp, # 新增
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )