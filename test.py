import argparse
import os.path as osp
import shutil
import tempfile

import cv2
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector

from model.VGG_models import VGG_models


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()),
                                  dtype=torch.uint8,
                                  device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config',default='configs/mask_rcnn_r50_fpn_1x.py', help='test config file path')
    parser.add_argument('checkpoint', default='model/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth',help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def calcRotation(outputs_c1x£¬outputs_c1y,fov)
{
	//the angle that need to rotate to vertical with x axis
	anglex = outputs_c1x * 2 * M_PI / panoWidth;
	angley = M_PI / 2 - outputs_c1y * M_PI / panoHeight;

	Rotz = np.zeros((3, 3, 1), np.uint8)
	Roty = np.zeros((3, 3, 1), np.uint8)
	Rot = np.zeros((3, 3, 1), np.uint8)

	Rotz(0, 0) = cos(_anglex);
	Rotz(0, 1) = sin(_anglex);
	Rotz(1, 0) = -sin(_anglex);
	Rotz(1, 1) = cos(_anglex);
	Roty(0, 0) = cos(_angley);
	Roty(0, 2) = sin(_angley);
	Roty(2, 0) = -sin(_angley);
	Roty(2, 2) = cos(_angley);
	Rot = Roty*Rotz;
	
	invRot = Rot.t();
	return invRot
}




def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   imgs_per_gpu=1,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')


    outputs = multi_gpu_test(model, data_loader, args.tmpdir)

	vggmodel = VGG()
	vggmodel.load_state_dict(torch.load('~/.torch/models/vgg16-397923af.pth'))
	vggmodel.cuda()
	_, vgg2deepfea = vggmodel(outputs)

	c1model = rankmodel()
	c1model.load_state_dict(torch.load('model/c1.pth'))
	c1gmodel.cuda()
	_, c1deep = rankmodel(vgg2deepfea)

	outputs_c1x = [out[x] for out in outputs]
	outputs_c1y = [out[y] for out in outputs]
	fov = 75 * M_PI / 180;
	invRot=calcRotation(outputs_c1x,outputs_c1y,fov)

	vggmodel = VGG()
	vggmodel.load_state_dict(torch.load('~/.torch/models/vgg16-397923af.pth'))
	vggmodel.cuda()
	_, c2deepfea = vggmodel(invRot)

	c2model = rankmodel()
	c2model.load_state_dict(torch.load('model/c2.pth'))
	c2gmodel.cuda()
	_, c2deep = rankmodel(c2deepfea)

	model = RF_Model(c2deep)
    model.load_state_dict(torch.load('RF.pth'))
	model.cuda()
	_, res = model(c2deep)

	save_path = './results/'
	res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    misc.imsave(save_path+name, res)


if __name__ == '__main__':
    main()
