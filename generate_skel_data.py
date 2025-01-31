"""
Run a forward pass of the SKEL model with default parameters (T pose) and export the resulting meshes.
Author: Marilyn Keller
"""

import os
import torch
from skel.skel_model import SKEL
from skel.kin_skel import *
import trimesh
import math
import ocifs
from composer.utils import dist, get_device
from argparse import ArgumentParser
import numpy as np
import pyrender
from PIL import Image
import pickle
from streaming import MDSWriter
from tqdm import tqdm

remote = "oci://mosaicml-internal-datasets/mosaicml-internal-dataset-multi-image/synthetic-anatomy-val"

columns = {
    'images': 'bytes',
    'messages': 'json',
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_generations", type=int, default=50000)
    parser.add_argument("--dist_timeout", type=float, default=300.0)
    parser.add_argument(
        "--sample_offset", type=int, default=0
    )
    args = parser.parse_args()


    print('getting device')
    device = get_device()
    print('initting dist')
    dist.initialize_dist(device, args.dist_timeout)
    print('making writer')
    writer = MDSWriter(out = f'{remote}/rank{dist.get_global_rank()}', compression = "zstd", columns = columns)
    device = f'cuda:{dist.get_local_rank()}'


    samples_per_rank, remainder = divmod(args.num_generations, dist.get_world_size())

    start_idx = dist.get_global_rank() * samples_per_rank + min(remainder, dist.get_global_rank())
    end_idx = start_idx + samples_per_rank
    if dist.get_global_rank() < remainder:
        end_idx += 1
    
    for sample_id in tqdm(range(start_idx, end_idx)):
        gender = 'female' if np.random.rand() > 0.5 else 'male'
        print(gender)

        correct = np.random.rand() > 0.5
        wrong_pose = list(pose_limits.keys())[np.random.randint(len(pose_limits.keys()))]
        skel = SKEL(gender=gender).to(device)

        # Set parameters to default values (T pose)
        pose = torch.zeros(1, skel.num_q_params).to(device) # (1, 46)
        betas = torch.zeros(1, skel.num_betas).to(device) # (1, 10)
        trans = torch.zeros(1, 3).to(device)

        for i, pose_name in enumerate(pose_param_names):
            if pose_name not in pose_limits:
                pose[0, i] = np.pi/2*np.random.rand() - np.pi/4
            else:
                if correct or pose_name != wrong_pose:
                    pose[0, i] = (pose_limits[pose_name][1] - pose_limits[pose_name][0]) * np.random.rand() + pose_limits[pose_name][0]
                else:
                    new_pose = np.random.rand()*2*np.pi - np.pi
                    while new_pose >= pose_limits[pose_name][0] and new_pose <= pose_limits[pose_name][1]:
                        new_pose = np.random.rand()*2*np.pi - np.pi
                    pose[0, i] = new_pose

        # SKEL forward pass
        skel_output = skel(pose, betas, trans)
        skin_trimesh = trimesh.Trimesh(vertices=skel_output.skin_verts.detach().cpu().numpy()[0], faces=skel.skin_f.cpu()) 
        print('trimesh generated')
        mesh = pyrender.Mesh.from_trimesh(skin_trimesh)
        scene = pyrender.Scene()
        scene.add(mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2)/2
        camera_pose = np.array([
            [1.0, 0,   0,   0],
            [0,  1.0, 0.0, -0.3],
            [0.0,  0.0,   1.0,   2.5],
            [0.0,  0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)

        light = pyrender.SpotLight(color=np.array([1.0, 1.0, 1.0]), intensity=11.0,
                                innerConeAngle=np.pi/40.0,
                                outerConeAngle=np.pi/3.0)
        scene.add(light, pose=camera_pose)
        r = pyrender.OffscreenRenderer(1000, 1000)

        print('rendered')

        color, depth = r.render(scene)

        output_image = Image.fromarray(color)
        output_imgs = pickle.dumps([output_image.convert("RGB")])

        messages = []
        messages.append({'role': 'user', 'content': f'<image>\n Does this image appear anatomically correct?'})
        if correct:
            messages.append({'role': 'assistant', 'content': 'Yes, this image is anatomically correct.'})
        else:
            messages.append({'role': 'assistant', 'content': f'No, this image is not anatomically correct, as there is an unnatural angle in {wrong_pose.replace("_", " ")}, where r and l signify left and right.'})
        
        writer.write({'images': output_imgs, 'messages': messages})
    
    writer.finish()
