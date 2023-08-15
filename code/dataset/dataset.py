#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
from itertools import chain
from glob import glob
import torch
import torch.nn.functional as Func
import numpy as np
import cv2
import trimesh
import open3d as o3d
from tqdm import tqdm

from utils import io, geometry

class EpochSampler:
    def __init__(self, reg_arrays):
        assert all([arr.shape[0] == reg_arrays[0].shape[0] for arr in reg_arrays])
        self.reg_arrays = reg_arrays
        self.num_total = reg_arrays[0].shape[0]
        self._rand_perm()
    def _rand_perm(self):
        self.rand_indexes = torch.randperm(self.num_total)
        self.index_in_rand = 0
    def sample(self, num_samples):
        start = self.index_in_rand
        self.index_in_rand += num_samples
        end = self.index_in_rand
        rand_indexes_slice = self.rand_indexes[start:end]

        sampled_arrays = [arr[rand_indexes_slice] for arr in self.reg_arrays]

        # permute all pixels only after traversing all
        if self.index_in_rand >= self.num_total:
            self._rand_perm()

        return sampled_arrays

class NeILFDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 data_folder,
                 validation_indexes,
                 num_pixel_samples,
                 required,
                 mode='train',
                 rescale=-1,
                 **kwargs
                 ):

        assert os.path.exists(data_folder), "Data directory is empty"
        print ('NeILFDataset: loading data from: ' + data_folder)

        self.data_folder = data_folder
        self.num_pixel_samples = num_pixel_samples
        self.mode = mode
        self.use_brdf_gt = False
        self.use_depth_map = False
        self.required = required

        # load cameras
        self.intrinsics, self.extrinsics, self.scale_mat, self.image_list, self.image_indexes, self.image_resolution = \
            io.load_cams_from_sfmscene(f'{self.data_folder}/inputs/sfm_scene.json')
        self.total_pixels = self.image_resolution[0] * self.image_resolution[1]

        # adjust scale mat
        def get_scaled_cams(ext, scale_mat):
            cam_centers = np.stack([(- v[:3,:3].T @ v[:3, 3:])[:,0] for v in ext.values()], axis=0)
            cam_centers = np.concatenate([cam_centers, np.ones_like(cam_centers[:,:1])], axis=-1)[:,:,None]
            scaled_centers = np.linalg.inv(scale_mat)[None,:,:] @ cam_centers
            scaled_centers = scaled_centers[:,:3,0]
            dists = np.linalg.norm(scaled_centers, axis=-1)
            return scaled_centers, dists

        _, dists = get_scaled_cams(self.extrinsics, self.scale_mat)
        self.max_scaled_cam_dist = dists.max()
        print(f'max scaled cam dist: {self.max_scaled_cam_dist}')

        if rescale != -1 and dists.max() > rescale:
            rad_scale = rescale / dists.max()
            self.scale_mat[range(3),range(3)] /= rad_scale
            print(f'radius scale: {rad_scale}')
            _, dists = get_scaled_cams(self.extrinsics, self.scale_mat)
            print(f'max rescaled cam dist: {dists.max()}')

        self.inv_scale_mat = np.linalg.inv(self.scale_mat)

        # split training/validataion sets
        self.num_images = len(self.image_indexes)
        validation_list_indexes = [v % self.num_images for v in validation_indexes]
        self.validation_indexes = []
        self.training_indexes = []
        for i in range(self.num_images):
            image_index = self.image_indexes[i]
            if i in validation_list_indexes:
                self.validation_indexes.append(image_index)
            else:
                self.training_indexes.append(image_index)
        self.num_validation_images = len(self.validation_indexes)
        self.num_training_images = len(self.training_indexes)

        # uv coordinate
        uv = np.mgrid[0:self.image_resolution[0], 0:self.image_resolution[1]]
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        self.uv = uv.reshape(2, -1).transpose(1, 0)                                             # [HW, 2]

        # load and prepare validation data
        self._load_and_prepare_validation_data()

        # load and prepare training data
        if mode == 'train':
            self._load_and_prepare_training_data()

        # load positional texture atlas for exporting BRDF as texture maps
        if mode == 'eval':
            texture_folder = os.path.join(self.data_folder, 'inputs/model/pos_tex')
            pos_tex_files = glob(os.path.join(texture_folder, '*_position.exr'))
            self.tex_coords = []
            for pos_tex_file in pos_tex_files:
                coord, coord_mask = io.load_tex_coord(pos_tex_file)                             # [H, W, 3], [H, W]
                coord_shape = coord.shape[:2]
                coord = torch.from_numpy(coord[coord_mask]).float()                             # [N_v, 3]
                # convert Blender coordinate system to ours
                # coord = torch.stack([coord[:, 0], -coord[:, 2], coord[:, 1]], dim=1)            # {x, -z, y} -> {x, y, z}     
                # normalize the coordinate system
                inv_scale_mat = torch.from_numpy(self.inv_scale_mat).float()                    # [4, 4]
                inv_scale_mat = inv_scale_mat.unsqueeze(0)                                      # [1, 4, 4]
                homo_coord = geometry.append_hom(coord, -1).unsqueeze(-1)                       # [N_v, 4, 1]
                coord = (inv_scale_mat @ homo_coord)[:, :3, 0]                                  # [N_v, 3]
                obj_name = (os.path.splitext(os.path.basename(pos_tex_file))[0]).split('_')[0]
                self.tex_coords.append((obj_name, coord, coord_shape, coord_mask))
        
        # ray trace
        if self.required['mesh']:
            mesh = trimesh.load(os.path.join(self.data_folder, 'inputs/model/mesh.obj'))
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            # adjust xyz inconsistency between blender and ours
            vertices = np.stack([vertices[:,0], -vertices[:,2], vertices[:,1]], axis=1)
            # apply scale mat
            scaled_vertices = np.expand_dims(np.linalg.inv(self.scale_mat), 0) @ np.expand_dims(geometry.append_hom(vertices, -1), -1)
            scaled_vertices = scaled_vertices[:,:3,0]
            self.tracer_mesh = (scaled_vertices, faces)

        # point cloud
        if self.required['pcd']:
            self.num_pcd_samples = kwargs['num_pcd_samples']
            if os.path.exists(opcd_path := os.path.join(self.data_folder, 'inputs/model/oriented_pcd.ply')):
                pcd, _ = self._load_pcd(opcd_path)
            elif os.path.exists(pcd_path := os.path.join(self.data_folder, 'inputs/model/pcd.ply')):
                print('use point cloud without normal')
                pcd, _ = self._load_pcd(pcd_path)
            else:
                raise FileNotFoundError('point cloud file not found')
            self.pcd_sampler = EpochSampler([pcd])
    
    def get_validation_data(self, downsample=None):
        if downsample is not None:
            sample, ground_truth = self.validation_data
            # sample['intrinsics'] = geometry.scale_intrinsic(sample['intrinsics'], 1/downsample)
            H = self.image_resolution[0] // downsample
            W = self.image_resolution[1] // downsample
            # sample['uv'] = sample['uv'].reshape(len(self.validation_indexes), *self.image_resolution, 2)[:,:H,:W,:]
            resized = {}
            for k, v in chain(sample.items(), ground_truth.items()):
                if k in ['intrinsics', 'pose']:
                    resized[k] = v
                else:
                    assert v.shape[0] == len(self.validation_indexes)
                    if v.shape[1] == self.total_pixels:
                        flatten = True
                        channel_shape = v.shape[2:]
                    elif v.shape[1] == self.image_resolution[0] and v.shape[2] == self.image_resolution[1]:
                        flatten = False
                        channel_shape = v.shape[3:]
                    else:
                        raise ValueError(f'Value with key {k} is not a map, please add it to the exclusion list.')
                    v = v.reshape(len(self.validation_indexes), *self.image_resolution, -1)  # nHWc
                    v = Func.interpolate(v.permute(0,3,1,2), (H, W), mode='bilinear', align_corners=False).permute(0,2,3,1)  # nhwc
                    shape = [len(self.validation_indexes)] + ([H*W] if flatten else [H, W]) + list(channel_shape)
                    resized[k] = v.reshape(*shape)
            sample = {k:resized[k] for k in sample}
            ground_truth = {k:resized[k] for k in ground_truth}
            return sample, ground_truth
        else:
            return self.validation_data

    def _load_and_prepare_validation_data(self):
        
        print ('NeILFDataset: loading validation data')

        # load validation views
        (
            list_rgb_pixels,
            list_position_pixels,
            list_normal_pixels,
            list_list_index_pixels,
            list_rgb_grad_pixels,
            list_basecolor_maps,
            list_roughness_maps,
            list_metallic_maps,
        ) = self._load_data_from_indexes(self.validation_indexes)

        # prepare validation data
        self.validation_data = []
        for list_index in range(self.num_validation_images):

            validation_sample = {
                'intrinsics': self.all_intrinsics[list_index],                                  # [1, 4, 4]
                'pose': self.all_poses[list_index],                                             # [1, 4, 4]
                'uv': self.uv,                                                                  # [HW, 2]
            }
            if self.required['position']: validation_sample['positions'] = list_position_pixels[list_index].squeeze(0)  # [HW, 3]
            if self.required['normal']: validation_sample['normals'] = list_normal_pixels[list_index].squeeze(0)        # [HW, 3]

            validation_ground_truth = {
                'rgb': list_rgb_pixels[list_index].squeeze(0)                                   # [HW, 3]
            }
            if self.use_brdf_gt:
                validation_ground_truth['base_color'] = list_basecolor_maps[list_index]         # [H, W, 3]        
                validation_ground_truth['roughness'] = list_roughness_maps[list_index]          # [H, W, 1]
                validation_ground_truth['metallic'] = list_metallic_maps[list_index]            # [H, W, 1]

            self.validation_data.append( (validation_sample, validation_ground_truth) )
        self.validation_data = self.collate_fn(self.validation_data)

    def _load_and_prepare_training_data(self):

        print ('NeILFDataset: loading training data')

        # load validation views
        (
            list_rgb_pixels,
            list_position_pixels,
            list_normal_pixels,
            list_list_index_pixels,
            list_rgb_grad_pixels,
            list_basecolor_maps,
            list_roughness_maps,
            list_metallic_maps,
        ) = self._load_data_from_indexes(self.training_indexes)

        # prepare training data by flatten to 1D lists         
        self.all_intrinsics = torch.cat(self.all_intrinsics, axis=0)                            # [T, 4, 4]
        self.all_poses = torch.cat(self.all_poses, axis=0)                                      # [T, 4, 4]

        self.train_rgb_pixels = torch.cat(list_rgb_pixels, axis=0)                              # [T, HW, 3]
        self.train_rgb_grad_pixels = torch.cat(list_rgb_grad_pixels, axis=0)                    # [T, HW]
        if self.required['position']: self.train_position_pixels = torch.cat(list_position_pixels, axis=0)      # [T, HW, 3]
        if self.required['normal']: self.train_normal_pixels = torch.cat(list_normal_pixels, axis=0)            # [T, HW, 3]
        self.train_list_index_pixels = torch.cat(list_list_index_pixels, axis=0)                # [T, HW]
        self.train_rgb_pixels = self.train_rgb_pixels.reshape([-1, 3])                          # [THW, 3]
        self.train_rgb_grad_pixels = self.train_rgb_grad_pixels.reshape([-1])                   # [THW]
        if self.required['position']: self.train_position_pixels = self.train_position_pixels.reshape([-1, 3])  # [THW, 3]
        if self.required['normal']: self.train_normal_pixels = self.train_normal_pixels.reshape([-1, 3])        # [THW, 3]
        self.train_list_index_pixels = self.train_list_index_pixels.reshape([-1])               # [THW]
        self.train_uv = self.uv.repeat([self.num_training_images, 1])                           # [THW, 2]

        self.num_all_training_pixels = (self.train_list_index_pixels).shape[0]

        # permute all training pixels
        DUMMY = self.train_list_index_pixels
        self.pixel_sampler = EpochSampler([
            self.train_rgb_pixels,
            self.train_rgb_grad_pixels,
            self.train_position_pixels  if self.required['position'] else DUMMY,
            self.train_normal_pixels    if self.required['normal'] else DUMMY,
            self.train_uv,
            self.train_list_index_pixels,
        ])

        # calculate median of rgb
        self.train_rgb_median = self.train_rgb_pixels.median().item()

    def _load_data_from_indexes(self, indexes_to_read):

        self.all_intrinsics = []
        self.all_poses = []
        list_rgb_pixels = []
        list_position_pixels = []
        list_normal_pixels = []
        list_list_index_pixels = []
        list_rgb_grad_pixels = []
        list_basecolor_maps = []
        list_roughness_maps = []
        list_metallic_maps = []
        
        # load all pixels
        for list_index, image_index in enumerate(tqdm(indexes_to_read)):

            # paths
            prefix = os.path.split(os.path.splitext(self.image_list[image_index])[0])[1]
            input_folder = os.path.join(self.data_folder, 'inputs')

            # only use depth map if there are depth maps but no position maps
            if not os.path.exists(os.path.join(input_folder, 'position_maps')) \
                and os.path.exists(os.path.join(input_folder, 'depth_maps')):
                geometry_type = 'depth_maps'
                self.use_depth_map = True

            geometry_type = 'depth_maps' if self.use_depth_map else 'position_maps'
            rgb_image_prefix = os.path.join(input_folder, 'images', prefix)
            position_map_prefix = os.path.join(input_folder, geometry_type, prefix)
            normal_map_prefix = os.path.join(input_folder, 'normal_maps', prefix)

            gt_brdf_folder = os.path.join(self.data_folder, 'ground_truths/materials')
            basecolor_map_path = os.path.join(gt_brdf_folder, 'kd', prefix + '.png')
            roughness_map_path = os.path.join(gt_brdf_folder, 'roughness', prefix + '.png')
            metallic_map_path = os.path.join(gt_brdf_folder, 'metallic', prefix + '.png')

            # only use brdf gt if there are brdf gts and in eval mode
            if os.path.isdir(os.path.join(gt_brdf_folder, 'kd')) and \
                os.path.isdir(os.path.join(gt_brdf_folder, 'roughness')) and \
                os.path.isdir(os.path.join(gt_brdf_folder, 'metallic')) and \
                self.mode == 'eval':
                self.use_brdf_gt = True

            # read input images, depth/position maps, and normal maps
            rgb_image = io.load_rgb_image_with_prefix(rgb_image_prefix)                         # [H, W, 3]
            rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)                              # [H, W]
            rgb_grad_x = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR)         # [H, W]
            rgb_grad_y = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=cv2.FILTER_SCHARR)         # [H, W]
            rgb_grad = cv2.magnitude(rgb_grad_x, rgb_grad_y)                                    # [H, W]
            if not self.required['position']: pass
            elif self.use_depth_map:
                depth_map = io.load_gray_image_with_prefix(position_map_prefix)                 # [H, W]
                position_map = geometry.depth_map_to_position_map(
                    depth_map, self.uv, 
                    self.extrinsics[image_index], self.intrinsics[image_index])                 # [H, W, 3], [H, W, 1]
            else:
                position_map = io.load_rgb_image_with_prefix(position_map_prefix)               # [H, W, 3]
            if self.required['normal']: normal_map = io.load_rgb_image_with_prefix(normal_map_prefix)  # [H, W, 3]

            # read BRDF ground truth for evaluation
            if self.use_brdf_gt:
                base_color = io.load_rgb_image(basecolor_map_path)                              # [H, W, 3]
                roughness = io.load_gray_image(roughness_map_path)                              # [H, W]
                metallic = io.load_gray_image(metallic_map_path)                                # [H, W]

            # apply scale mat to camera
            projection = self.intrinsics[image_index] @ self.extrinsics[image_index]            # [4, 4]
            scaled_projection = (projection @ self.scale_mat)[0:3, 0:4]                         # [3, 4]    
            intrinsic, pose = geometry.decompose_projection_matrix(scaled_projection)           # [4, 4], [4, 4]
            self.all_intrinsics.append(torch.from_numpy(intrinsic).float().unsqueeze(0))        # [N][1, 4, 4]             
            self.all_poses.append(torch.from_numpy(pose).float().unsqueeze(0))                  # [N][1, 4, 4]

            # apply scale mat to position map
            if self.required['position']:
                mask_map = (position_map != 0).astype('float32')                                    # [H, W, 1]
                mask_map = np.sum(mask_map, axis=-1, keepdims=True)                                 # [H, W, 1]
                mask_map = (mask_map > 0).astype('float32')                                         # [H, W, 1]
                position_map = geometry.append_hom(torch.from_numpy(position_map).float(), -1)      # [H, W, 3] -> [H, W, 4]  
                position_map = position_map.unsqueeze(-1)                                           # [H, W, 4, 1]
                inv_scale_mat = torch.from_numpy(self.inv_scale_mat).float()                        # [4, 4]
                inv_scale_mat = inv_scale_mat.unsqueeze(0).unsqueeze(0)                             # [1, 1, 4, 4]
                position_map = (inv_scale_mat @ position_map)[:, :, 0:3, 0]
                mask_map = torch.from_numpy(mask_map).float()                                       # [H, W, 1]
                position_map = position_map * mask_map                                              # [H, W, 3]

            # to tensors and reshape to pixels
            rgb_pixels = torch.from_numpy(rgb_image.reshape(1, -1, 3)).float()                  # [1, HW, 3]
            rgb_grad_pixels = torch.from_numpy(rgb_grad.reshape(1, -1)).float()                 # [1, HW]
            if self.required['position']: position_pixels = position_map.reshape(1, -1, 3)                      # [1, HW, 3]  
            if self.required['normal']: normal_pixels = torch.from_numpy(normal_map.reshape(1, -1, 3)).float()  # [1, HW, 3]
            list_index_pixels = torch.ones_like(rgb_pixels[0:1, :, 0]) * list_index             # [1, HW]
            
            # to tensors
            if self.use_brdf_gt:
                basecolor_pixels = torch.from_numpy(base_color).float()                         # [H, W, 3]
                roughness_pixels = torch.from_numpy(roughness).float()                          # [H, W, 1]
                metallic_pixels = torch.from_numpy(metallic).float()                            # [H, W, 1]

            # add to list 
            list_rgb_pixels.append(rgb_pixels)                                                  # [N][1, HW, 3]
            list_rgb_grad_pixels.append(rgb_grad_pixels)                                        # [N][1, HW]
            if self.required['position']: list_position_pixels.append(position_pixels)          # [N][1, HW, 3]
            if self.required['normal']: list_normal_pixels.append(normal_pixels)                # [N][1, HW, 3]
            list_list_index_pixels.append(list_index_pixels)                                    # [N][1, HW]
            if self.use_brdf_gt:
                list_basecolor_maps.append(basecolor_pixels)                                    # [N][H, W, 3]
                list_roughness_maps.append(roughness_pixels)                                    # [N][H, W]
                list_metallic_maps.append(metallic_pixels)                                      # [N][H, W]
            
        return (
            list_rgb_pixels,
            list_position_pixels,
            list_normal_pixels,
            list_list_index_pixels,
            list_rgb_grad_pixels,
            list_basecolor_maps,
            list_roughness_maps,
            list_metallic_maps,
        )

    def _load_pcd(self, path, downsample_target=-1):
        if type(path) is str:
            fused_pcd_o3d = o3d.io.read_point_cloud(path)
        else:
            sampled_points, sampled_index = trimesh.sample.sample_surface(path, 1000000)  # NOTE hard code
            sampled_normals = path.face_normals[sampled_index]
            fused_pcd_o3d = o3d.geometry.PointCloud()
            fused_pcd_o3d.points = o3d.utility.Vector3dVector(sampled_points)
            fused_pcd_o3d.normals = o3d.utility.Vector3dVector(sampled_normals)
        normalized_o3d = fused_pcd_o3d.transform(self.inv_scale_mat)
        bbox = o3d.geometry.TriangleMesh.create_box(2,2,2).translate(np.array([-1,-1,-1])).get_axis_aligned_bounding_box()
        cropped_o3d = normalized_o3d.crop(bbox)
        fused_points = torch.from_numpy(np.asarray(cropped_o3d.points)).float()
        has_normal = cropped_o3d.has_normals()
        if has_normal:
            fused_normals = torch.from_numpy(np.asarray(cropped_o3d.normals)).float()
            fused_normals = Func.normalize(fused_normals, dim=-1)
            fused_pcd = torch.cat([fused_points, fused_normals], dim=-1)
        else:
            fused_pcd = fused_points

        if downsample_target != -1 and fused_pcd.shape[0] > downsample_target:
            downsampled_o3d = normalized_o3d.random_down_sample(downsample_target / fused_pcd.shape[0])
            fused_points_d = torch.from_numpy(np.asarray(downsampled_o3d.points)).float()
            if has_normal:
                fused_normals_d = torch.from_numpy(np.asarray(downsampled_o3d.normals)).float()
                fused_normals_d = Func.normalize(fused_normals_d, dim=-1)
                downsampled_pcd = torch.cat([fused_points_d, fused_normals_d], dim=-1)
            else:
                downsampled_pcd = fused_points_d
        else:
            downsampled_pcd = fused_pcd

        return fused_pcd, downsampled_pcd

    def __len__(self):
        # unbounded length
        return 10000000             

    def __getitem__(self, index_not_used):

        [
            rgb_batch,                                                                          # [B, 3]
            rgb_grad_batch,                                                                     # [B]
            position_batch,                                                                     # [B, 3]
            normal_batch,                                                                       # [B, 3]
            uv_batch,                                                                           # [B, 2]
            index_batch,                                                                        # [B]
        ] = self.pixel_sampler.sample(self.num_pixel_samples)
        if self.required['pcd']: pcd_batch = self.pcd_sampler.sample(self.num_pcd_samples)[0]
        index_batch = index_batch.long()
        intrinsic_batch = self.all_intrinsics[index_batch]                                      # [B, 4, 4]
        pose_batch = self.all_poses[index_batch]                                                # [B, 4, 4]

        sample = {
            'intrinsics': intrinsic_batch,
            'pose': pose_batch,
            'uv': uv_batch,
        }
        if self.required['position']: sample['positions'] = position_batch
        if self.required['normal']: sample['normals'] = normal_batch
        if self.required['pcd']: sample['pcd_positions'] = pcd_batch[..., :3]

        ground_truth = {
            'rgb': rgb_batch,
            'rgb_grad': rgb_grad_batch,
        }
        if self.required['pcd'] and pcd_batch.shape[-1] == 6: ground_truth['pcd_normals'] = pcd_batch[..., 3:]
        
        return sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)