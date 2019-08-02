# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-27 01:06:36
# @Last Modified by:   yulidong
# @Last Modified time: 2019-08-02 22:10:07

""" Training perception and control """
import argparse
from os.path import join, exists
from os import mkdir
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from models.vae import VAE
from models.action_vae import VAE_a
from models.controller import Controller
import visdom
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=64*8, metavar='N',
					help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
					help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', default='log',type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
					help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
					help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()
learning_rate=1e-3

torch.manual_seed(111)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")


transform_train = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((RED_SIZE, RED_SIZE)),
	# transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
])

transform_test = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((RED_SIZE, RED_SIZE)),
	transforms.ToTensor(),
])


trained=0
#model = VAE(3, LSIZE).to(device)
model=VAE(3, LSIZE)
model=torch.nn.DataParallel(model,device_ids=range(8))
model.cuda()
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))
model_p=VAE_a(7, LSIZE)
model_p=torch.nn.DataParallel(model_p,device_ids=range(8))
model_p.cuda()
optimizer_p = optim.Adam(model_p.parameters(),lr=learning_rate,betas=(0.9,0.999))
controller=Controller(LSIZE,3)
controller=torch.nn.DataParallel(controller,device_ids=range(8))
controller=controller.cuda()
optimizer_a = optim.SGD(controller.parameters(),lr=learning_rate*10)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
# earlystopping = EarlyStopping('min', patience=30)

vis = visdom.Visdom(env='corner_train')

current_window = vis.image(
	np.random.rand(64, 64),
	opts=dict(title='current!', caption='current.'),
)
recon_window = vis.image(
	np.random.rand(64, 64),
	opts=dict(title='Reconstruction!', caption='Reconstruction.'),
)
mask_window = vis.image(
	np.random.rand(64, 64),
	opts=dict(title='mask!', caption='mask.'),
)
future_window = vis.image(
	np.random.rand(64, 64),
	opts=dict(title='future!', caption='future.'),
)
pre_window = vis.image(
	np.random.rand(64, 64),
	opts=dict(title='prediction!', caption='prediction.'),
)
loss_window = vis.line(X=torch.zeros((1,)).cpu(),
					   Y=torch.zeros((1)).cpu(),
					   opts=dict(xlabel='minibatches',
								 ylabel='Loss',
								 title='Training Loss',
								 legend=['Loss']))
lossc_window = vis.line(X=torch.zeros((1,)).cpu(),
					   Y=torch.zeros((1)).cpu(),
					   opts=dict(xlabel='minibatches',
								 ylabel='Loss',
								 title='Reconstruction Loss',
								 legend=['Reconstruction Loss']))
lossa_window = vis.line(X=torch.zeros((1,)).cpu(),
					   Y=torch.zeros((1)).cpu(),
					   opts=dict(xlabel='minibatches',
								 ylabel='Loss',
								 title='controller Loss',
								 legend=['controller Loss']))
lossp_window = vis.line(X=torch.zeros((1,)).cpu(),
					   Y=torch.zeros((1)).cpu(),
					   opts=dict(xlabel='minibatches',
								 ylabel='Loss',
								 title='prediction Loss',
								 legend=['prediction Loss']))

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, sigma):
	""" VAE loss function """
	#BCE = torch.mean(torch.sum(torch.pow(recon_x-x,2),dim=(1,2,3)))
	mask= torch.sum(x,dim=1,keepdim=True)
	mask= (mask<torch.mean(mask.view(mask.shape[0],1,-1))).float().cuda()
	mask_distance=mask
	#mask_distance[:,:,-16:,:]=mask_distance[:,:,-16:,:]*10
	#mask_distance[:,:,-32:-16,:]=mask_distance[:,:,-32:-16,:]*5
	#mask_distance[:,:,-48:-32,:]=mask_distance[:,:,-48:-32,:]*2
	# BCE = F.mse_loss(recon_x*mask_distance,x*mask_distance,reduction='sum')/torch.sum(mask)+ \
	# 	0.1*F.mse_loss(recon_x*(1-mask),x*(1-mask),reduction='sum')/torch.sum(1-mask)
	BCE = F.mse_loss(recon_x*mask_distance,x*mask_distance,reduction='sum')+ \
		0.1*F.mse_loss(recon_x*(1-mask),x*(1-mask),reduction='sum')
	BCE=BCE/x.shape[0]
	#print(torch.mean(recon_x).item())
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	sigma=torch.max(sigma,1e-4*torch.ones_like(sigma).cuda())
	#print(torch.mean(torch.pow(sigma,2)).item(),torch.mean(torch.pow(mu,2)).item())
	KLD = -0.5 * torch.sum(1 + torch.log(torch.pow(sigma,2)) - torch.pow(mu,2) - torch.pow(sigma,2),dim=1)
	#KLD=torch.max(KLD,torch.ones_like(KLD).cuda()*LSIZE*0.5)
	KLD=torch.mean(KLD)
	# print(KLD.shape,sigma.shape,mu.shape)
	# exit()
	print(BCE.item(),KLD.item())
	return BCE + KLD


def train(epoch,vae_dir,training_sample):
	""" One training epoch """
	model.train()
	model_p.train()
	controller.train()
	train_loss = []
	for batch_idx, [data,action,pre] in enumerate(train_loader):
		#torch.autograd.set_detect_anomaly(True)
		data = data.cuda()
		action=action.cuda()
		pre=pre.cuda()
		optimizer.zero_grad()
		optimizer_p.zero_grad()
		optimizer_a.zero_grad()
		recon_c, mu_c, var_c = model(data)
		loss_c = loss_function(recon_c, data, mu_c, var_c)
		recon_f, mu_f, var_f = model(pre)
		loss_f = loss_function(recon_f, pre, mu_f, var_f)
		recon_p, mu_p, var_p = model_p(torch.cat([data,action],dim=1))
		loss_p = loss_function(recon_p, pre, mu_p, var_p)
		mu, sigma = mu_c.detach().cuda(), var_c.detach().cuda()
		#sigma = torch.exp(sigma/2.0)
		epsilon = torch.randn_like(sigma)
		z=mu+sigma*epsilon
		z=z.cuda().view(data.shape[0],-1).detach()
		action_p=controller(z)
		#print(action[:,:,0,0])
		loss_a=F.mse_loss(action_p,action[:,:3,11,11],reduction='mean')
		#action_pr=torch.cat([action_p.detach().view(action_p.shape[0],3,1,1).expand(action_p.shape[0],3,action.shape[-2],action.shape[-1]),action[:,2:3,...]])
		action_pr=action_p.detach().view(action_p.shape[0],3,1,1).expand(action_p.shape[0],3,action.shape[-1],action.shape[-2])
		action_pr=torch.cat([action_pr,action[:,2:3,:,:]],dim=1)
		recon_pr, mu_pr, var_pr = model_p(torch.cat([data,action_pr],dim=1))
		loss_pr = loss_function(recon_pr, pre, mu_pr, var_pr)

		loss=loss_c+loss_f+loss_p+loss_a+loss_pr
		if torch.isnan(loss) or torch.isinf(loss):
			print('nan or inf error:',loss.item() )
			continue
		loss.backward()

		#print(loss.item())
		train_loss.append(loss.item())
		optimizer.step()
		optimizer_p.step()
		optimizer_a.step()
		ground = data[0,...].data.cpu().numpy().astype('float32')
		ground = np.reshape(ground, [3,64, 64])
		vis.image(
			ground,
			opts=dict(title='ground!', caption='ground.'),
			win=current_window,
		)
		image = recon_c[0,...].data.cpu().numpy().astype('float32')
		image = np.reshape(image, [3, 64, 64])
		vis.image(
			image,
			opts=dict(title='Reconstruction!', caption='Reconstruction.'),
			win=recon_window,
		)
		image=np.sum(ground,axis=0)
		image=(image<np.mean(image)).astype('float32')
		vis.image(
			image,
			opts=dict(title='Reconstruction!', caption='Reconstruction.'),
			win=mask_window,
		)
		ground = pre[0,...].data.cpu().numpy().astype('float32')
		ground = np.reshape(ground, [3,64, 64])
		vis.image(
			ground,
			opts=dict(title='future!', caption='ground.'),
			win=future_window,
		)
		image = recon_p[0,...].data.cpu().numpy().astype('float32')
		image = np.reshape(image, [3, 64, 64])
		vis.image(
			image,
			opts=dict(title='prediction!', caption='prediction.'),
			win=pre_window,
		)
		# if loss.item()>5:
		# 	loss=loss/loss
		# if loss_c>5:
		# 	loss_c=loss_c/loss_c
		# if loss_a>5:
		# 	loss_a=loss_c/loss_a
		# if loss_c>5:
		# 	loss_p=loss_c/loss_p
		vis.line(
			X=torch.ones(1).cpu() *training_sample,
			Y=loss.item() * torch.ones(1).cpu(),
			win=loss_window,
			update='append')
		vis.line(
			X=torch.ones(1).cpu() * training_sample,
			Y=loss_c.item() * torch.ones(1).cpu(),
			win=lossc_window,
			update='append')
		vis.line(
			X=torch.ones(1).cpu() * training_sample,
			Y=loss_a.item() * torch.ones(1).cpu(),
			win=lossa_window,
			update='append')
		vis.line(
			X=torch.ones(1).cpu() * training_sample,
			Y=loss_p.item() * torch.ones(1).cpu(),
			win=lossp_window,
			update='append')
		training_sample+=1
		if batch_idx % 1 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}% training_sample:{:.0f}]  Loss_c: {:.4f}  Loss_f: {:.4f}  Loss_p: {:.4f}  Loss_a: {:.4f}  Loss_pr: {:.4f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				len(data) * batch_idx / len(train_loader)/10,training_sample,
				loss_c.item(),loss_f.item(),loss_p.item(),loss_a.item(),loss_pr.item()))
		if (batch_idx+1)%1000==0:
			best_filename = join(vae_dir, 'best.pkl')
			filename_vae = join(vae_dir, 'vae_checkpoint_'+str(epoch)+'.pkl')
			filename_pre = join(vae_dir, 'pre_checkpoint_'+str(epoch)+'.pkl')
			filename_control = join(vae_dir, 'contorl_checkpoint_'+str(epoch)+'.pkl')
			# is_best = not cur_best or test_loss < cur_best
			# if is_best:
			#	 cur_best = test_loss
			is_best=False
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),

			}, is_best, filename_vae, best_filename)
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model_p.state_dict(),
				'optimizer': optimizer_p.state_dict(),

			}, is_best, filename_pre, best_filename)
			save_checkpoint({
				'epoch': epoch,
				'state_dict': controller.state_dict(),
				'optimizer': optimizer_a.state_dict(),

			}, is_best, filename_control, best_filename)
	return training_sample


def test():
	""" One test epoch """
	model.eval()
	test_loss = []
	with torch.no_grad():
		for batch_idx, [data,action,pre] in enumerate(train_loader):
			data = data.cuda()
			recon_batch, mu, var = model(data)
			test_loss.append(loss_function(recon_batch, data, mu, var).item())
			ground = data[0, ...].data.cpu().numpy().astype('float32')
			ground = np.reshape(ground, [3, 64, 64])
			vis.image(
				ground,
				opts=dict(title='ground!', caption='ground.'),
				win=current_window,
			)
			image = recon_batch[0,...].data.cpu().numpy().astype('float32')
			image = np.reshape(image, [3, 64, 64])
			vis.image(
				image,
				opts=dict(title='image!', caption='image.'),
				win=recon_window,
			)
	test_loss =np.mean(test_loss)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
	mkdir(vae_dir)
	mkdir(join(vae_dir, 'samples'))

# reload_file = join(vae_dir, 'best.pkl')
# if not args.noreload and exists(reload_file):
#	 state = torch.load(reload_file)
#	 print("Reloading model at epoch {}"
#		   ", with test error {}".format(
#			   state['epoch'],
#			   state['precision']))
#	 model.load_state_dict(state['state_dict'])
#	 optimizer.load_state_dict(state['optimizer'])
#	 trained=state['epoch']
	#trained=0
	# scheduler.load_state_dict(state['scheduler'])
	# earlystopping.load_state_dict(state['earlystopping'])
state = torch.load('/home/ld/gym-car/log/vae/contorl_checkpoint_52.pkl')
controller.load_state_dict(state['state_dict'])
optimizer_a.load_state_dict(state['optimizer'])
print('contorller load success')
state = torch.load('/home/ld/gym-car/log/vae/pre_checkpoint_52.pkl')
model_p.load_state_dict(state['state_dict'])
optimizer_p.load_state_dict(state['optimizer'])
print('prediction load success')
state = torch.load('/home/ld/gym-car/log/vae/vae_checkpoint_52.pkl')
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
trained=state['epoch']
print('vae load success')
trained=0
cur_best = None
all_data=6000
sample_data=1000
sample_buff=all_data/sample_data
sample_count=0
training_sample=0
for epoch in range(trained+1, args.epochs + 1):
	dataset_train = RolloutObservationDataset('/data/result/',transform_train, train=True,sample_data=sample_data,sample_count=sample_count)
	#dataset_test = RolloutObservationDataset('/data/result/',transform_test, train=False,sample_data=sample_data,sample_count=sample_count)
	sample_count+=1
	if sample_count==sample_buff:
		sample_count=0
	train_loader = torch.utils.data.DataLoader(
		dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=32,drop_last=True)
	# test_loader = torch.utils.data.DataLoader(
	# 	dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=32,drop_last=True)
	training_sample=train(epoch,vae_dir,training_sample)
	#exit()
	#test_loss = test()
	# scheduler.step(test_loss)
	# earlystopping.step(test_loss)

	# checkpointing
	best_filename = join(vae_dir, 'best.pkl')
	filename_vae = join(vae_dir, 'vae_checkpoint_'+str(epoch)+'.pkl')
	filename_pre = join(vae_dir, 'pre_checkpoint_'+str(epoch)+'.pkl')
	filename_control = join(vae_dir, 'contorl_checkpoint_'+str(epoch)+'.pkl')
	# is_best = not cur_best or test_loss < cur_best
	# if is_best:
	#	 cur_best = test_loss
	is_best=False
	# save_checkpoint({
	# 	'epoch': epoch,
	# 	'state_dict': model.state_dict(),
	# 	'optimizer': optimizer.state_dict(),

	# }, is_best, filename_vae, best_filename)
	# save_checkpoint({
	# 	'epoch': epoch,
	# 	'state_dict': model_p.state_dict(),
	# 	'optimizer': optimizer_p.state_dict(),

	# }, is_best, filename_pre, best_filename)
	# save_checkpoint({
	# 	'epoch': epoch,
	# 	'state_dict': controller.state_dict(),
	# 	'optimizer': optimizer_a.state_dict(),

	# }, is_best, filename_control, best_filename)


	if not args.nosamples:
		print('saving image')
		with torch.no_grad():
			sample = torch.randn(RED_SIZE, LSIZE).cuda()
			sample = model.module.decoder(sample).cpu()
			save_image(np.reshape(sample,[RED_SIZE, 3, RED_SIZE, RED_SIZE]),
					   join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

	# if earlystopping.stop:
	#	 print("End of Training because of early stopping at epoch {}".format(epoch))
	#	 break
