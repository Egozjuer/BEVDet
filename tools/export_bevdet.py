# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import sys

# sys.path.remove('/home/zjlab/fq/perception/mmlab/mmdetection3d')
# sys.path.append('/home/zjlab/fq/perception/od/BEVDet')
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
import sys
sys.path.append("./")

# #            "args":["--config", "configs/bevdet/bevdet-sttiny.py",
#                 "--checkpoint","weights/bevdet-sttiny-pure.pth",

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path',default ="/home/zjlab/fq/perception/od/BEVDet/configs/bevdet/bevdet-sttiny.py")
    parser.add_argument('--checkpoint', help='checkpoint file',default = '/home/zjlab/fq/perception/od/BEVDet/weights/bevdet-sttiny-pure.pth')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    # if args.eval and args.format_only:
    #     raise ValueError('--eval and --format_only cannot be both specified')

    # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    #     raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))


    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    device='cuda:0'

    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    # export onnx
    # x = torch.zeros(6,3,256,704,device = 'cuda:0')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=model.to(device)
            self.x1 = torch.randn(1,6,3,3,device=device)
            self.x2 =  torch.randn(1,6,3,device=device)
            self.x3 = torch.randn(1,6,3,3,device=device)
            self.x4 = torch.randn(1,6,3,3,device=device)
            self.x5 = torch.randn(1,6,3,device=device)            

        def forward(self,x):
            ib, ic, ih, iw = map(int, x.shape)
            x = self.model.img_backbone(x)
            x = self.model.img_neck(x)
            x = x.view(1, 6, 512, 16, 44)
            x6=[self.x1,self.x2,self.x3,self.x4,self.x5] 
            x = self.model.img_view_transformer([x]+x6)
            x = self.model.img_bev_encoder_backbone(x)
            x = self.model.img_bev_encoder_neck(x)#result x torch.Size([1, 256, 128, 128])
            x = self.model.pts_bbox_head([x])

            return x[0]

    m=Model().eval()
    image=torch.randn(6,3,256,704,device = device)
    torch_out=torch.onnx.export(
        m, image, "./BEVdet.onnx",
        export_params=True,
        input_names=["images"],
        # output_names=["output1","ouput2","ouput3","ouput4","ouput5","ouput6"],
        output_names=["output1"],
        opset_version=16)


        # dynamic_axes={
        #     "images": {0: "batch"},
        #     "output": {0: "batch"}
        # }
    
    print("Done.!")
    # model.eval()
    # x1 = torch.randn(1,6,3,3)
    # x2 =  torch.randn(1,6,3)
    # x3 = torch.randn(1,6,3,3)
    # x4 = torch.randn(1,6,3,3)
    # x5 = torch.randn(1,6,3)
    # x6 =[x1,x2,x3,x4,x5]
    # x = torch.randn(6,3,256,704)
    # x = model.img_backbone(x)
    # x = model.img_neck(x)
    # x = x.view(1, 6, 512, 16, 44)

    # x = model.img_view_transformer([x]+x6)
    # x = model.img_bev_encoder_backbone(x)
    # x = model.img_bev_encoder_neck(x)#result x torch.Size([1, 256, 128, 128])
    # x = model.pts_bbox_head([x])
    # print("infer end:")

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('fork')
    main()
