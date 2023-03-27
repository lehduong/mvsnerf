import sys,os,imageio
root = '/home/duongl/mvsnerf'
os.chdir(root)
sys.path.append(root)

from opt import config_parser
from data import dataset_dict
# models
from models import *
from renderer import *
from data.ray_utils import get_rays
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from data.ray_utils import ray_marcher

torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs


def unpreprocess(data, shape=(1,1,3,1,1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
    return (data - mean) / std


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    print('center in poses_avg', center)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * N_rots, N+1)[:-1]:
        c = np.array([(np.cos(theta)*theta)/10, (-np.sin(theta)*theta)/10, -0.1]) 
        z = -(normalize(c - np.array([0,0,-focal])))
        print("c", c)
        print("z", z)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_far, rads_scale=0.5, N_views=120):

    # center pose
    c2w = poses_avg(c2ws_all)
    print('poses_avg', c2w)
    
    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    print('near and far bounds', close_depth, inf_depth)
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    print(focal)

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:,:3,3] - c2w[:3,3][None]
    rads = np.percentile(np.abs(tt), 70, 0)*rads_scale
    print("rads",rads)
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


def pose_spherical_nerf(euler, radius=0.01):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,-radius])
    return c2ws_render

def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)

def nerf_video_path(c2ws, theta_range=10,phi_range=20,N_views=120):
    c2ws = torch.tensor(c2ws)
    mean_position = torch.mean(c2ws[:,:3, 3],dim=0).reshape(1,3).cpu().numpy()
    rotvec = []
    for i in range(c2ws.shape[0]):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
    rotvec = np.mean(np.stack(rotvec), axis=0)
    render_poses = [pose_spherical_nerf(rotvec+np.array([angle,0.0,-phi_range])) for angle in np.linspace(-theta_range,theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([theta_range,0.0,angle])) for angle in np.linspace(-phi_range,phi_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([angle,0.0,phi_range])) for angle in np.linspace(theta_range,-theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([-theta_range,0.0,angle])) for angle in np.linspace(phi_range,-phi_range,N_views//4, endpoint=False)]
    return render_poses


for i_scene, scene in enumerate(['0d251b65dfa04e3e88cd8774b77c3a27']):#'ship','drums','ficus','materials',

    cmd = f'--datadir objaverse/{scene} \
     --dataset_name blender --white_bkgd --imgScale_test {1.0} '

    is_finetued = True # set True if rendering with finetuning
    if is_finetued:
        cmd += f'--ckpt runs_fine_tuning/{scene}/ckpts/latest.tar'
        pad = 0 #the padding value should be same as your finetuning ckpt
    else:
        cmd += '--ckpt ./ckpts/mvsnerf-v0.tar'
        pad = 24 #the padding value should be same as your finetuning ckpt
        
    args = config_parser(cmd.split())
    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim = 8+3*4

    if i_scene==0 or is_finetued:
        render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(render_kwargs_train)

        MVSNet = render_kwargs_train['network_mvs']
        render_kwargs_train.pop('network_mvs')


    datadir = args.datadir
    datatype = 'val'
    args.chunk = 5120
    frames = 30
    dataset = dataset_dict[args.dataset_name](args, split=datatype)
    val_idx = torch.tensor(np.arange(3)).cuda() #dataset.img_idx
    
    save_as_image = False
    save_dir = f'results/video2'
    os.makedirs(save_dir, exist_ok=True)
    MVSNet.train()
    MVSNet = MVSNet.cuda()
    
    with torch.no_grad():

        if is_finetued:   
            # large baselien
            c2ws_all = dataset.load_poses_all()
            random_selete = torch.randint(0,len(c2ws_all),(1,))     #!!!!!!!!!! you may change this line if rendering a specify view 
            dis = np.sum(c2ws_all[:,:3,2] * c2ws_all[[random_selete],:3,2], axis=-1)
            pair_idx = np.argsort(dis)[::-1][torch.randperm(15)[:3]]
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
            volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
            volume_feature = RefVolume(volume_feature.detach()).cuda()
            # c2ws_render = nerf_video_path(pose_source['c2ws'].cpu(), N_views=frames)
            # c2ws_render = nerf_video_path(pose_source['c2ws'].cpu(), theta_range=60, phi_range=120, N_views=frames)
            c2ws_render = gen_render_path(c2ws_all[pair_idx], N_views=frames)
            c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)
        else:            
            # neighboring views with angle distance
            c2ws_all = dataset.load_poses_all()
            random_selete = torch.randint(0,len(c2ws_all),(1,))     #!!!!!!!!!! you may change this line if rendering a specify view 
            dis = np.sum(c2ws_all[:,:3,2] * c2ws_all[[random_selete],:3,2], axis=-1)
            pair_idx = np.argsort(dis)[::-1][torch.randperm(5)[:3]]
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)#pair_idx=pair_idx, 
            volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
            
            #####
            c2ws_render = gen_render_path(c2ws_all[pair_idx], N_views=frames)
            c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)

            
        imgs_source = unpreprocess(imgs_source)
        

        try:
            tqdm._instances.clear() 
        except Exception:     
            pass
        
        frames = []
        img_directions = dataset.directions.to(device)
        for i, c2w in enumerate(tqdm(c2ws_render)):
            torch.cuda.empty_cache()
            
            rays_o, rays_d = get_rays(img_directions.cuda(), torch.tensor(c2w, dtype=torch.float32).cuda())  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d,
                     near_far_source[0] * torch.ones_like(rays_o[:, :1]),
                     near_far_source[1] * torch.ones_like(rays_o[:, :1])],
                    1).to(device)  # (H*W, 3)
            
        
            N_rays_all = rays.shape[0]
            rgb_rays, depth_rays_preds = [],[]
            for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                    N_samples=args.N_samples)

                # Converting world coordinate to ndc coordinate
                H, W = imgs_source.shape[-2:]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)


                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
                                                                       xyz_NDC, z_vals, rays_o, rays_d,
                                                                       volume_feature,imgs_source, **render_kwargs_train)
    
                
                rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
                rgb_rays.append(rgb)
                depth_rays_preds.append(depth_pred)

            
            depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
            depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
            
            rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
            H_crop, W_crop = np.array(rgb_rays.shape[:2])//20
            img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)

            frames.append(img_vis.astype('uint8'))
    imageio.mimwrite(f'{save_dir}/ft_{scene}_spiral.mp4', np.stack(frames), fps=10, quality=10)