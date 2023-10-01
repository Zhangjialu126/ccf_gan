''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
from pytorch_fid.fid_score import calculate_fid_given_paths
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datasets as dset
from utils import CenterCropLongEdge

def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    G = model.Generator(**config).cuda()
    utils.count_parameters(G)
    if config['recons']:
        D = model.Discriminator(**config).cuda()

    # Load weights
    print('Loading weights...')
    # Here is where we deal with the ema--load ema weights or load normal weights
    utils.load_weights(G if not (config['use_ema']) else None, D if config['recons'] else None, state_dict,
                       config['weights_root'], experiment_name, config['load_weights'],
                       G if config['ema'] and config['use_ema'] else None,
                       strict=False, load_optim=False)
    # Update batch size setting used for G
    G_batch_size = config['batch_size']  # max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'],
                               z_var=config['z_var'])
    if config['recons']:
        val_loaders = utils.get_data_loaders(**{**config, 'batch_size': config['batch_size'],
                                                'start_itr': state_dict['itr']})
        sample_rec = functools.partial(utils.sample_rec, y_=y_,
                                       G=G,
                                       D=D, loader=val_loaders[0], device=device, config=config)

    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()
    else:
        print('G is in %s mode...' % ('training' if G.training else 'eval'))

    # Sample function
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    if config['accumulate_stats']:
        print('Accumulating standing stats across %d accumulations...' % config['num_standing_accumulations'])
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Sample a number of images and save them to an NPZ, for use with TF-Inception
    if config['sample_npz']:
        # convinient root dataset
        root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5',
                     'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
                     'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
                     'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
                     'C10': 'cifar', 'C100': 'cifar',
                     'L64': 'lsun', 'L128': 'lsun', 'A64': 'celeba',
                     'A128': 'celeba', 'I_tiny': 'tiny-imagenet-200', 'VGGFace2': 'VGG-Face2',
                     'V200': 'VGG-Face200', 'V500': 'VGG-Face500', 'V1000': 'VGG-Face1000'}
        dset_dict = {'I32': dset.ImageFolder, 'I64': dset.ImageFolder,
                     'I128': dset.ImageFolder, 'I256': dset.ImageFolder,
                     'I32_hdf5': dset.ILSVRC_HDF5, 'I64_hdf5': dset.ILSVRC_HDF5,
                     'I128_hdf5': dset.ILSVRC_HDF5, 'I256_hdf5': dset.ILSVRC_HDF5,
                     'C10': dset.CIFAR10, 'C100': dset.CIFAR100,
                     'L64': torchvision.datasets.LSUN, 'L128': torchvision.datasets.LSUN,
                     'A64': dset.ImageFolder, 'A128': dset.ImageFolder,
                     'I_tiny': dset.ImageFolder, 'VGGFace2':dset.ImageFolder,
                     'V200': dset.ImageFolder, 'V500': dset.ImageFolder, 'V1000': dset.ImageFolder}
        imsize_dict = {'I32': 32, 'I32_hdf5': 32,
                       'I64': 64, 'I64_hdf5': 64,
                       'I128': 128, 'I128_hdf5': 128,
                       'I256': 256, 'I256_hdf5': 256,
                       'C10': 32, 'C100': 32,
                       'L64': 64, 'L128': 128, 'A64': 64,
                       'A128': 128, 'I_tiny': 64, 'VGGFace2': 64,
                       'V200': 64, 'V500': 64, 'V1000': 64}
        # We first sample real images
        x = []
        print('Sampling %d generated images and saving them to npz...' % config['sample_num_npz'])
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        data_root = config['data_root'] + '/' + root_dict[config['dataset']]
        which_dataset = dset_dict[config['dataset']]
        if config['dataset'] in ['C10', 'C100']:
            sample_transform = []
        elif config['dataset'] in ['L64', 'L128', 'A64', 'A128']:
            sample_transform = [transforms.Resize(imsize_dict[config['dataset']]), transforms.CenterCrop(imsize_dict[config['dataset']])]
            if config['dataset'] in ['L64', 'L128']:
                dataset_kwargs = {'classes': ['bedroom_train']}
        else:
            sample_transform = [CenterCropLongEdge(), transforms.Resize(imsize_dict[config['dataset']])]
        sample_transform = transforms.Compose(sample_transform + [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
        dataset_kwargs = {'index_filename': '%s_imgs.npz' % config['dataset']}
        sample_set = which_dataset(root=data_root, transform=sample_transform, **dataset_kwargs)
        sample_loader = DataLoader(sample_set, batch_size=G_batch_size,
                                   shuffle=True, num_workers=16)
        for i, data in enumerate(sample_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            print(i)
            images, _ = data
            x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
            if i > int(np.ceil(config['sample_num_npz'] / float(G_batch_size))):
                break
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        x = x.transpose((0, 2, 3, 1))
        real_npz_filename = '%s/%s/reals.npz' % (config['samples_root'], experiment_name)
        print('Saving npz to %s...' % real_npz_filename)
        np.savez(real_npz_filename, **{'x': x})

        # Now we sample fake images
        x = []
        print('Sampling %d generated images and saving them to npz...' % config['sample_num_npz'])
        for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
            with torch.no_grad():
                images, _ = sample()
                x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        x = x.transpose((0, 2, 3, 1))
        fake_npz_filename = '%s/%s/samples.npz' % (config['samples_root'], experiment_name)
        print('Saving npz to %s...' % fake_npz_filename)
        np.savez(fake_npz_filename, **{'x': x})

        # calculate tf FIDs
        paths = []
        paths += [real_npz_filename]
        paths += [fake_npz_filename]
        # paths += [config['data_root'] + '/' + root_dict[config['dataset']]]
        # paths += ['I128_inception_moments.npz']
        print('Now calculating fids ...')
        tf_fid = calculate_fid_given_paths(paths, batch_size=500)
        print('Pytorch FID is: ', tf_fid)

        # Lists to hold images and labels for images
        if config['recons']:
            x = []
            print('Reconstructing %d images and saving them to npz...' % config['sample_num_npz'])
            for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
                with torch.no_grad():
                    images, _ = sample_rec()
                    x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
            x = np.concatenate(x, 0)[:config['sample_num_npz']]
            x = x.transpose((0, 2, 3, 1))
            npz_filename = '%s/%s/rec_samples.npz' % (config['samples_root'], experiment_name)
            print('Saving recons npz to %s...' % npz_filename)
            np.savez(npz_filename, **{'x': x})
            paths = []
            paths += [npz_filename]
            paths += [config['data_root'] + '/' + root_dict[config['dataset']]]
            tf_fid = calculate_fid_given_paths(paths, None)
            print('Tnsorflow FID for reconstruction is: ', tf_fid)
    # Prepare sample sheets
    if config['sample_sheets']:
        print('Preparing conditional sample sheets...')
        utils.sample_sheet(G, classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                           num_classes=config['n_classes'],
                           samples_per_class=10, parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=config['sample_sheet_folder_num'],
                           z_=z_, )
    # Sample interp sheets
    if config['sample_interps']:
        print('Preparing interp sheets...')
        for fix_z, fix_y in zip([False, False, True], [False, True, False]):
            utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
                               num_classes=config['n_classes'],
                               parallel=config['parallel'],
                               samples_root=config['samples_root'],
                               experiment_name=experiment_name,
                               folder_number=config['sample_sheet_folder_num'],
                               sheet_number=0,
                               fix_z=fix_z, fix_y=fix_y, device='cuda')
    # Sample random sheet
    if config['sample_random']:
        print('Preparing random sample sheet...')
        images, labels = sample()
        torchvision.utils.save_image(images.float(),
                                     '%s/%s/random_samples.jpg' % (config['samples_root'], experiment_name),
                                     nrow=int(G_batch_size ** 0.5),
                                     normalize=True)

    # Get Inception Score and FID
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'],
                                                                      config['no_fid'])

    # Prepare a simple function get metrics that we use for trunc curves
    def get_metrics():
        sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
        IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10,
                                                     prints=False, use_torch=False)
        # Prepare output string
        outstring = 'Using %s weights ' % ('ema' if config['use_ema'] else 'non-ema')
        outstring += 'in %s mode, ' % ('eval' if config['G_eval_mode'] else 'training')
        outstring += 'with noise variance %3.3f, ' % z_.var
        outstring += 'over %d images, ' % config['num_inception_images']
        if config['accumulate_stats'] or not config['G_eval_mode']:
            outstring += 'with batch size %d, ' % G_batch_size
        if config['accumulate_stats']:
            outstring += 'using %d standing stat accumulations, ' % config['num_standing_accumulations']
        outstring += 'Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
            state_dict['itr'], IS_mean, IS_std, FID)
        print(outstring)

    if config['sample_inception_metrics']:
        print('Calculating Inception metrics...')
        get_metrics()

    # # Sample truncation curve stuff. This is basically the same as the inception metrics code
    # if config['sample_trunc_curves']:
    #     start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
    #     print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
    #     for var in np.arange(start, end + step, step):
    #         z_.var = var
    #         # Optionally comment this out if you want to run with standing stats
    #         # accumulated at one z variance setting
    #         if config['accumulate_stats']:
    #             utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
    #                                             config['num_standing_accumulations'])
    #         get_metrics()


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
