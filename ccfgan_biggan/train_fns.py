''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import utils
import losses
import math


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}

    return train


def CFGAN_training_function(G, D, GD, cf_loss_fn, z_, y_, ema, state_dict, config, writer):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)


        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step

            # train t_nets
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                cf_x, cf_target_batch = GD(z_, y_,
                                    x[counter], y[counter], train_G=False,
                                    split_D=config['split_D'])

                critic_loss = cf_loss_fn(cf_x, cf_target_batch)
                D_loss = - critic_loss
                D_loss = D_loss / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # training generator/sampler
        # train gan loss
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            cf_x, cf_target_batch = GD(z_, y_,
                                       x[counter], y[counter], train_G=True,
                                       split_D=config['split_D'])
            # forward
            critic_loss = cf_loss_fn(cf_x, cf_target_batch)
            G_loss = critic_loss
            G_loss = G_loss / float(config['num_G_accumulations'])
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()) * 1e4,
               'D_loss': float(D_loss.item()) * 1e4}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss', D_loss.item(), state_dict['itr'])
        return out

    return train

def CFGAN_training_function_cond(G, D, GD, cf_loss_fn, z_, y_, ema, state_dict, config, writer):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0
        # y to embed
        y2embed = torch.arange(0, config['n_classes'], device='cuda')

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)


        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step

            # train t_nets
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                cf_x, cf_target, h_x, h_target = GD(z_, y_, x[counter], y[counter], train_G=False,
                                                                  split_D=config['split_D'])
                _, y_x_pre = D.classifier(h_x)
                y_target_pre, _ = D.classifier(h_target)
                y_x = y_
                y_target = y[counter]
                y_pre = torch.cat([y_x_pre, y_target_pre], 0)
                y_gt = torch.cat([y_x, y_target], 0)
                CELoss = nn.CrossEntropyLoss()
                C_loss_r = 0.05 * CELoss(y_target_pre, y_target) / float(config['num_D_accumulations'])
                C_loss_f = 0.05 * CELoss(y_x_pre, y_x) / float(config['num_D_accumulations'])
                C_loss_D = C_loss_r + C_loss_f

                y_embed, y_embed_out_ch = D.y_embedding(y2embed)

                Softmax = nn.Softmax(dim=1)
                y_x_pre = Softmax(y_x_pre)
                y_target_pre = Softmax(y_target_pre)
                critic_loss = cf_loss_fn(cf_x, cf_target, y_x_pre.detach(), y_target_pre.detach(), y_embed,
                                         y_embed_out_ch, config['ty_lambda'])
                D_loss = - critic_loss / float(config['num_D_accumulations'])

                total_loss = D_loss + C_loss_D
                total_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()


        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # training generator/sampler
        # train gan loss
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            cf_x, cf_target, h_x, h_target = GD(z_, y_, x[counter], y[counter], train_G=True, split_D=config['split_D'])

            y_x_pre, y_x_pre_f = D.classifier(h_x)
            y_target_pre, y_target_pre_f = D.classifier(h_target)
            y_x = y_
            y_target = y[counter]
            y_pre = torch.cat([y_x_pre, y_target_pre], 0)
            y_pre_f = torch.cat([y_x_pre_f, y_target_pre_f], 0)
            y_gt = torch.cat([y_x, y_target], 0)
            CELoss = nn.CrossEntropyLoss()
            C_loss_r = 0.5 * CELoss(y_pre, y_gt) / float(config['num_G_accumulations'])
            C_loss_f = 0.5 * CELoss(y_pre_f, y_gt) / float(config['num_G_accumulations'])
            C_loss_G = C_loss_r - C_loss_f

            y_embed, y_embed_out_ch = D.y_embedding(y2embed)

            Softmax = nn.Softmax(dim=1)
            y_x_pre = Softmax(y_x_pre)
            y_target_pre = Softmax(y_target_pre)
            critic_loss = cf_loss_fn(cf_x, cf_target, y_x_pre.detach(), y_target_pre.detach(), y_embed, y_embed_out_ch,
                                     config['ty_lambda'])
            G_loss = critic_loss / float(config['num_G_accumulations'])
            total_loss = G_loss + C_loss_G
            total_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])

        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()) * 1e4,
               'D_loss': float(D_loss.item()) * 1e4,
               'C_loss_D': float(C_loss_D.item()) * 1e4,
               'C_loss_G': float(C_loss_G.item()) * 1e4}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss', D_loss.item(), state_dict['itr'])
        writer.add_scalar('C_loss_D', C_loss_D.item(), state_dict['itr'])
        writer.add_scalar('C_loss_G', C_loss_G.item(), state_dict['itr'])
        return out

    return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config, writer):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                    x[counter], y[counter], train_G=False,
                                    split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        writer.add_scalar('G_loss', G_loss.item(), state_dict['itr'])
        writer.add_scalar('D_loss_real', D_loss_real.item(), state_dict['itr'])
        writer.add_scalar('D_loss_fake', D_loss_fake.item(), state_dict['itr'])
        return out

    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name, x=None):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')
    if config['which_train_fn'] == 'RCFGAN':
        utils.interpolated_imgs(G, D, x, num_of_groups=10, config=config, device='cuda',
                                samples_root=config['samples_root'],
                                experiment_name=experiment_name,
                                folder_number=state_dict['itr'], )


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''


def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log, writer):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,
                                                 config['num_inception_images'],
                                                 num_splits=10, use_torch=False)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
        state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID))
    writer.add_scalar('FID', FID, state_dict['itr'])
    writer.add_scalar('IS', IS_mean, state_dict['itr'])

''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''