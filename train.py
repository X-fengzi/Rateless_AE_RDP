import argparse
import json
from torchvision.utils import save_image
from torch.optim.lr_scheduler import LambdaLR
import models.base_model as base_model
from utils import *
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import math


def is_progress_interval(args, epoch):
    return epoch == args.n_epochs-1 or (args.progress_intervals > 0 and epoch % args.progress_intervals == 0)

def _lr_factor(epoch, dataset, mode=None):
    if dataset == 'mnist':
        if epoch < 20:
            return 1
        elif epoch < 40:
            return 1/5
        else:
            return 1/50
    elif dataset == 'fashion_mnist':
        if epoch < 20:
            return 1
        elif epoch < 35:
            return 1/5
        else:
            return 1/50
    elif dataset == 'svhn':
        if epoch < 25:
            return 1
        else:
            return 1/5
    else:
        return 1

def compute_lambda_anneal(Lambda, epoch, Lambda_init=0.0005, end_epoch=12):
    assert Lambda == 0 and epoch >= 0
    e = min(epoch, end_epoch)

    return Lambda_init*(end_epoch-e)/end_epoch

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Source: https://github.com/andreaferretti/wgan/blob/master/train.py
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def log_drop(x):
    return math.log(x+6)

def test(args,test_dataloader,encoder,decoder,discriminator,epoch,experiment_path,alpha1,SNR=None,device):

    with torch.no_grad():
            encoder.eval()
            decoder.eval()
            discriminator.eval()

            '''
            is_entropy_interval = args.entropy_intervals > 0 and (epoch < 5 or epoch % args.entropy_intervals == 0)
            if is_entropy_interval or ((epoch == args.n_epochs - 1 or epoch == 0) and args.entropy_intervals != -2):
                # use test batch size on training set for efficiency
                base_estimate_entropy_(encoder, args.latent_dim, args.L, args.Lambda_base, 'train',
                                       args.test_batch_size, experiment_path, args.dataset, device)
            '''

            for L in range(9):   
                distortion_loss_avg, perception_loss_avg = 0, 0                 
                for j, (x_test, y_test) in enumerate(test_dataloader):
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    u1_test = uniform_noise([x_test.size(0), args.latent_dim], alpha1).to(device)

                    n_row = 10
                    n_column = 10
                    n_images = n_row*n_column
                    # test_index = []
                    # for i in range(10):
                    #     for ii in range(x_test.size(0)):
                    #         # offset = torch.randint(low=0, high=x_test.size(0))
                    #         offset = 170
                    #         if y_test[(ii+offset)%x_test.size(0)] == i:
                    #             test_index.append((ii+offset)%x_test.size(0))
                    #             break
                    test_index = [71,265,186,382,163,162,361,34,110,16]
                    if j == 0:
                        plot_graph = x_test.data[test_index]

                    code = Dropout_rateless_rate(encoder(x_test, u1_test, y_test),p = L/9, is_tail=args.is_tail, device=device)
                    code = AWGN_channel(code,SNR,device)
                    # print(code[0])
                    x_test_recon = decoder(code, u1_test, y_test)
                    distortion_loss, perception_loss = evaluate_losses(x_test, x_test_recon, discriminator1)
                    distortion_loss_avg += x_test.size(0) * distortion_loss
                    perception_loss_avg += x_test.size(0) * perception_loss
                    if j == 0:
                        plot_graph = torch.cat([plot_graph,x_test_recon.data[test_index]],dim=0)  
                distortion_loss_avg /= test_set_size
                perception_loss_avg /= test_set_size
                with open(f'{experiment_path}/{SNR}dB/{L}_losses.csv', 'a') as f:
                    f.write(f'{epoch},{distortion_loss_avg},{perception_loss_avg}\n')                  
                    
            if is_progress_interval(args, epoch):
                # save_image(unnormalizer(plot_graph), f"{experiment_path}/{epoch}_recon.png", nrow=n_row, normalize=True)
                save_image(plot_graph, f"{experiment_path}/{SNR}dB/{epoch}_recon.png", nrow=n_row, normalize=True)
                print("the image has saved")
                if not saved_original_test_image:
                    save_image(unnormalizer(x_test.data[test_index]), f"{experiment_path}/{epoch}_real.png", nrow=n_row, normalize=True)
                    saved_original_test_image = True           

            encoder.train()
            decoder.train()
            discriminator.train()


def train(args, device):
    experiment_path = args.experiment_path
    if args.is_drop:
        experiment_path = experiment_path + '/drop'+'_'+str(args.latent_dim)
    if args.is_SAE:
        experiment_path = experiment_path + '/SAE'+'_'+str(args.latent_dim)

    experiment_path = experiment_path + args.set_path

    use_si = args.use_si

    assert (args.L > 0 or not args.quantize) and args.latent_dim > 0
    assert not (args.L > 0 and not args.quantize), f'Quantization disabled, yet args.L={args.L}'

    # Loss weight for gradient penalty
    lambda_gp = args.Lambda_gp

    # Initialize decoder and discriminator
    encoder = base_model.Encoder(args).to(device)
    decoder = base_model.Decoder(args).to(device)
    discriminator1 = base_model.Discriminator(args).to(device)
    alpha1 = encoder.alpha

    if args.initialize_mse_model:
        # Load pretrained models to continue from if directory is provided
        if args.Lambda_base > 0:
            assert isinstance(args.load_mse_model_path, str)

            # Check args match
            with open(os.path.join(args.load_mse_model_path, '_settings.json'), 'r') as f:
                mse_model_args = json.load(f)
                assert_args_match(mse_model_args, vars(args), ('L', 'latent_dim', 'limits', 'enc_layer_scale'))
                assert mse_model_args['Lambda_base'] == 0
                # No need to assert args match for "stochastic" and "quantize"?

        if isinstance(args.load_mse_model_path, str):
            assert args.Lambda_base > 0, args.load_mse_model_path
            encoder.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'encoder.ckpt')))
            decoder.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'decoder.ckpt')))
            discriminator1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'discriminator1.ckpt')))

    # Configure data loader
    train_dataloader, test_dataloader, unnormalizer = \
        load_dataset(args.dataset, args.batch_size, args.test_batch_size, shuffle_train=True)
    test_set_size = len(test_dataloader.dataset)

    # Optimizers
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder, betas=(args.beta1_encoder, args.beta2_encoder))
    optimizer_G = torch.optim.Adam(decoder.parameters(), lr=args.lr_decoder, betas=(args.beta1_decoder, args.beta2_decoder))
    optimizer_D = torch.optim.Adam(discriminator1.parameters(), lr=args.lr_critic, betas=(args.beta1_critic, args.beta2_critic))

    lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

    scheduler_E = LambdaLR(optimizer_E, lr_factor)
    scheduler_G = LambdaLR(optimizer_G, lr_factor)
    scheduler_D = LambdaLR(optimizer_D, lr_factor)

    criterion = nn.MSELoss()

    use_indirect = args.use_indirect
    save_pre = 'l='+str(args.Lambda_base)

    dim_str = 'd='+str(args.latent_dim)
    if use_indirect:
        dim_str+='-ind'

    os.makedirs(f"{experiment_path}", exist_ok=True)
    os.makedirs(f"{experiment_path+'/'+dim_str}", exist_ok=True)
    experiment_path_l = experiment_path+'/'+dim_str+'/'+save_pre
    os.makedirs(f"{experiment_path_l}", exist_ok=True)
    with open(f'{experiment_path}/_settings.json', 'w') as f:
        json.dump(vars(args), f)

    with open(f'{experiment_path_l}/_losses.csv', 'w') as f:
        f.write('epoch,distortion_loss,perception_loss\n')

    batches_done = 0
    n_cycles = 1 + args.n_critic
    disc_loss = torch.Tensor([-1])
    distortion_loss = torch.Tensor([-1])
    saved_original_test_image = False

    for epoch in range(args.n_epochs):
        Lambda = args.Lambda_base
        # if Lambda == 0:
        #     # Give an early edge to training discriminator for Lambda = 0
        #     Lambda = compute_lambda_anneal(Lambda, epoch)

        for i, (x, y) in enumerate(train_dataloader):
            # Configure input
            x = x.to(device)
            # x: [bsize, 1, 28, 28]
            y = y.to(device)
            # y: [bsize]

            if use_indirect:
                noise_a=0.5
                ind_noise = torch.randn(size=x.shape)*noise_a
                xn = x + ind_noise.cuda()
                x = xn

            if i % n_cycles != 1:
                #  Train Discriminator
                free_params(discriminator1)
                frozen_params(encoder)
                frozen_params(decoder)

                optimizer_D.zero_grad()

                # Noise batch_size x latent_dim
                cr = uniform_noise([x.size(0), args.latent_dim], alpha1).to(device)
                # cr: [b_size, latent_dim]

                code = encoder(x, cr, y)

                if args.is_drop:
                    code = Dropout_rateless(code, p = 0.5, mode = args.drop_mode, is_tail=args.is_tail, distribution=log_drop, power_beta=args.tail_drop_power_beta, device=device)
                if args.AWGN:
                    code = AWGN_channel(code,args.SNR,device)   
                x_recon = decoder(code, cr, y)
                # Real images
                real_validity = discriminator1(x)
                # Fake images
                fake_validity = discriminator1(x_recon)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator1, x.data, x_recon.data)
                # Adversarial loss
                disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                disc_loss.backward()

                optimizer_D.step()

            else:
                #  Train Generator
                frozen_params(discriminator1)
                free_params(encoder)
                free_params(decoder)

                optimizer_E.zero_grad()
                optimizer_G.zero_grad()

                cr = uniform_noise([x.size(0), args.latent_dim], alpha1).to(device)

                # loss = 0
                # for dim in range(args.latent_dim):

                #     code = encoder(x, cr, y)
                #     code = Dropout_rateless(code, p = 1-(dim+1)/args.latent_dim, mode = args.drop, is_tail=args.is_tail, distribution=log_drop)
                #     x_recon = decoder(code, cr, y)

                #     # real_validity = discriminator(x)
                #     fake_validity = discriminator1(x_recon)

                #     perception_loss = -torch.mean(fake_validity) # + torch.mean(real_validity)
                #     distortion_loss = criterion(x, x_recon)
                #     # if use_indirect:
                #     #     distortion_loss = criterion(xn, x_recon)

                #     loss += (dim+1)*(args.Lambda_distortion*distortion_loss + Lambda*perception_loss)/args.latent_dim

                # loss.backward()

                # optimizer_G.step()
                # optimizer_E.step()

                # drop = base_model.Dropout_rateless(p = 0.5)

                code = encoder(x, cr, y)

                loss_kld = 0
                if args.is_SAE:
                    tho_tensor = torch.FloatTensor([args.SAE_lamda for _ in range(args.latent_dim)]).unsqueeze(0).to(device)
                    code_avg = torch.sum(code, dim=0, keepdim=True) / code.shape[0]
                    loss_kld = kl_divergence(tho_tensor, code_avg)

                if args.is_drop:
                    code = Dropout_rateless(code, p = 0.5, mode = args.drop_mode, is_tail=args.is_tail, distribution=log_drop, power_beta=args.tail_drop_power_beta, device=device)
                if args.AWGN:
                    code = AWGN_channel(code,args.SNR,device)
                if i == 1:
                    print(code[0])
                x_recon = decoder(code, cr, y)

                real_validity = discriminator1(x)
                fake_validity = discriminator1(x_recon)

                perception_loss = -torch.mean(fake_validity) # + torch.mean(real_validity)
                distortion_loss = criterion(x, x_recon)
                # if use_indirect:
                #     distortion_loss = criterion(xn, x_recon)

              

                loss = args.Lambda_distortion*distortion_loss + Lambda*perception_loss + args.KL_loss_lamda*loss_kld
                loss.backward()

                optimizer_G.step()
                optimizer_E.step()

            if batches_done % 100 == 0:
                with torch.no_grad(): # use most recent results
                    real_validity = discriminator1(x)
                    perception_loss = -torch.mean(fake_validity) + torch.mean(real_validity)
                print(
                    "[Epoch %d/%d] [Batch %d/%d (batches_done: %d)] [Disc loss: %f] [Perception loss: %f] [Distortion loss: %f]"
                    % (epoch, args.n_epochs, i, len(train_dataloader), batches_done, disc_loss.item(),
                    perception_loss.item(), distortion_loss.item())
                )

            batches_done += 1

        # ---------------------
        # Evaluate losses on test set
        # ---------------------
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            discriminator1.eval()

            '''
            is_entropy_interval = args.entropy_intervals > 0 and (epoch < 5 or epoch % args.entropy_intervals == 0)
            if is_entropy_interval or ((epoch == args.n_epochs - 1 or epoch == 0) and args.entropy_intervals != -2):
                # use test batch size on training set for efficiency
                base_estimate_entropy_(encoder, args.latent_dim, args.L, args.Lambda_base, 'train',
                                       args.test_batch_size, experiment_path, args.dataset, device)
            '''

            distortion_loss_avg, perception_loss_avg = 0, 0
                     
            for j, (x_test, y_test) in enumerate(test_dataloader):
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                u1_test = uniform_noise([x_test.size(0), args.latent_dim], alpha1).to(device)

                n_row = 10
                n_column = 10
                n_images = n_row*n_column

                test_index = []
                for i in range(10):
                    for ii in range(x_test.size(0)):
                        # offset = torch.randint(low=0, high=x_test.size(0))
                        offset = 170
                        if y_test[(ii+offset)%x_test.size(0)] == i:
                            test_index.append((ii+offset)%x_test.size(0))
                            break
                test_index = [71,265,186,382,163,162,361,34,110,16]
                plot_graph = x_test.data[test_index]
                for L in range(9):  

                    code = Dropout_rateless_rate(encoder(x_test, u1_test, y_test),p = L/9, is_tail=args.is_tail, device=device)
                    # print(code[0])
                    x_test_recon = decoder(code, u1_test, y_test)
                    distortion_loss, perception_loss = evaluate_losses(x_test, x_test_recon, discriminator1)
                    distortion_loss_avg += x_test.size(0) * distortion_loss
                    perception_loss_avg += x_test.size(0) * perception_loss
                    plot_graph = torch.cat([plot_graph,x_test_recon.data[test_index]],dim=0)                       
                
                if j == 0 and is_progress_interval(args, epoch):
                    # save_image(unnormalizer(plot_graph), f"{experiment_path}/{epoch}_recon.png", nrow=n_row, normalize=True)
                    save_image(plot_graph, f"{experiment_path}/{epoch}_recon.png", nrow=n_row, normalize=True)
                    print("the image has saved")
                    if not saved_original_test_image:
                        save_image(unnormalizer(x_test.data[test_index]), f"{experiment_path}/{epoch}_real.png", nrow=n_row, normalize=True)
                        saved_original_test_image = True

            distortion_loss_avg /= test_set_size
            perception_loss_avg /= test_set_size

            with open(f'{experiment_path_l}/_losses.csv', 'a') as f:
                f.write(f'{epoch},{distortion_loss_avg},{perception_loss_avg}\n')

            encoder.train()
            decoder.train()
            discriminator1.train()

        scheduler_E.step()
        scheduler_D.step()
        scheduler_G.step()

    # ---------------------
    #  Save
    # ---------------------

    encoder1_file = f'{experiment_path}/encoder.ckpt'
    decoder1_file = f'{experiment_path}/decoder.ckpt'
    discriminator1_file = f'{experiment_path}/discriminator1.ckpt'

    torch.save(encoder.state_dict(), encoder1_file)
    torch.save(decoder.state_dict(), decoder1_file)
    torch.save(discriminator1.state_dict(), discriminator1_file)


if __name__ == '__main__':
    os.makedirs("experiments", exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_si", type=bool, default=False)
    parser.add_argument("--use_indirect", type=bool, default=False)
    parser.add_argument("--only_si", type=bool, default=False)

    parser.add_argument("--Lambda_base", type=float, default=0,
                        help="coefficient for perception loss for training base model (default: 0.0)")
    parser.add_argument("--latent_dim", type=int, default=24, help="dimensionality of the latent space")

    parser.add_argument("--is_drop", type=bool, default=True, help="the distribution of drop out")
    parser.add_argument("--drop_mode", type=str, default="uniform", help="the distribution of drop out")
    parser.add_argument("--is_tail", type=bool, default=True, help="the method of drop out")
    parser.add_argument("--tail_drop_power_beta", type=float, default=0.67, help="the value of power beta")

    parser.add_argument("--is_SAE", type=bool, default=False, help="if sparse AE")
    parser.add_argument("--SAE_lamda", type=float, default=0.1, help="the sparse degree of SAE")
    parser.add_argument("--KL_loss_lamda", type=float, default=1, help="the coeff of KL loss in SAE")

    parser.add_argument("--AWGN", type=bool, default=True, help="add AWGN channel")
    parser.add_argument("--SNR", type=float, default=5, help="the SNR(dB) of channel")

    parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--n_channel", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument('--quantize', type=int, default=1, help='Whether to quantize or not. [Does nothing!] Always active.')
    parser.add_argument('--stochastic', type=int, default=1, help='add noise below quantization threshold (default: True)')
    parser.add_argument('--L', type=int, default=4, help='number of quantization levels for base model (default: -1)')
    parser.add_argument('--limits', nargs=2, type=float, default=[-1,1], help='quanitzation limits (default: (-1,1))')
    parser.add_argument("--Lambda_distortion", type=float, default=1.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_joint", type=float, default=1.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_gp", type=float, default=10.0, help="coefficient for gradient penalty")
    parser.add_argument("--lr_encoder", type=float, default=1e-2, help="encoder learning rate")
    parser.add_argument("--lr_decoder", type=float, default=1e-2, help="decoder learning rate")
    parser.add_argument("--lr_critic", type=float, default=2e-4, help="critic learning rate")
    parser.add_argument("--beta1_encoder", type=float, default=0.5, help="encoder beta 1")
    parser.add_argument("--beta1_decoder", type=float, default=0.5, help="decoder beta 1")
    parser.add_argument("--beta1_critic", type=float, default=0.5, help="critic beta 1")
    parser.add_argument("--beta2_encoder", type=float, default=0.9, help="encoder beta 2")
    parser.add_argument("--beta2_decoder", type=float, default=0.9, help="decoder beta 2")
    parser.add_argument("--beta2_critic", type=float, default=0.9, help="critic beta 2")
    parser.add_argument("--test_batch_size", type=int, default=5000, help="test set batch size (default: 5000)")
    parser.add_argument("--load_mse_model_path", type=str, default=None, help="directory from which to preload enc1/dec1+disc1 models to start training at")
    parser.add_argument("--load_base_model_path", type=str, default=None, help="directory from which to preload enc1/dec1+disc1 models to start training at")
    parser.add_argument("--initialize_base_discriminator", type=int, default=0, help="For refined or reduced models: whether to start from base model disc.")
    parser.add_argument("--initialize_mse_model", type=int, default=0, help="For base model: whether or not to continue training from Lambda=0 model.")
    parser.add_argument("--enc_layer_scale", type=float, default=1.0, help="Scale layer size of encoder by factor")
    parser.add_argument("--reduced_dims", type=str, default='', help="Reduced dims")
    parser.add_argument("--dataset", type=str, default='mnist', help="dataset to use (default: mnist)")
    parser.add_argument("--progress_intervals", type=int, default=1, help="periodically show progress of training")
    parser.add_argument("--entropy_intervals", type=int, default=-1, help="periodically calculate entropy of model. -1 only end, -2 for never")
    parser.add_argument("--submode", type=str, default=None, help="generic submode of mode")
    parser.add_argument("--mode", type=str, default='base', help="base, refined or reduced training mode")
    parser.add_argument("--experiment_path", type=str, default='./wrl/experiments', help="name of the subdirectory to save")
    parser.add_argument("--set_path", type=str, default='_power_0.67_AWGN_5dB', help="name of the subdirectory to save")
    

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device]: {device}')

    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        vars(args)['input_size'] = 784
        vars(args)['n_class'] = 10
    elif args.dataset == 'svhn':
        vars(args)['input_size'] = 3*32*32
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    if args.mode == 'base':
        train(args, device)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')
