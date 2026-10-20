import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from models.health import HealthLog

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, health_log=None,
                 test_ema=False, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        
        self.optimizers.append(self.optG)
        lambda1 = lambda epoch: 1 if epoch < 100 else 1 - (epoch - 100) / (200 - 100)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optG, lr_lambda=lambda1)
        self.resume_training() 
        # The LR schedule isn't in self.schedulers, so it isn't saved and a resumed run would restart
        # it at epoch 0. Continue it from the resumed epoch instead.
        if self.epoch > 0:
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.lr_scheduler.step()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)
        # ctp_only_eval_plan.md §4: the test phase samples with the EMA copy (decisions #22). The
        # copy was made before netG got its noise schedule, so it needs its own. Training is untouched.
        self.test_ema = bool(test_ema) and self.phase == 'test'
        if self.test_ema:
            if self.ema_scheduler is None:
                raise ValueError('test_ema needs ema_scheduler: there is no EMA copy to test')
            ema_net = self.netG_EMA.module if self.opt['distributed'] else self.netG_EMA
            ema_net.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        # ctp_only_plan.md §16: per-epoch and per-monitor-pass health files, rank 0 only
        self.health = None
        if health_log is not None and self.opt['global_rank'] == 0:
            self.health = HealthLog(self.opt['path']['experiments_root'], **health_log)
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image')).type(torch.float32)
        self.gt_image = self.set_device(data.get('gt_image')).type(torch.float32)
        self.mask = self.set_device(data.get('mask'))
        # None unless the dataset sends one; then the loss counts only it (ctp_only_plan.md §7.2)
        self.loss_mask = self.set_device(data.get('loss_mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {

            'gt_image': self.gt_image.detach()[:].float().cpu(),
            'cond_image': self.cond_image.detach()[:].float().cpu(),
        }
        if self.task in ['my_mission_mask','my_mission_mask_ddim']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': self.mask_image,
            })
        if phase != 'train':

            dict.update({
                'output': self.output.detach()[:].float().cpu()
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.output[idx].detach().float().cpu())
        if self.task in ['my_mission_mask','my_mission_mask_ddim']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        if self.health is not None:
            self.health.start_epoch()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask, loss_mask=self.loss_mask)
            loss.backward()
            if self.health is not None:
                self.health.step(self.netG, loss, self.batch_size)
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
        if self.health is not None:
            self.health.end_epoch(self.epoch, self.iter, self.optG.param_groups[0]['lr'], self.lr_scheduler.last_epoch)
        self.lr_scheduler.step()
        print(self.lr_scheduler.get_last_lr())

        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        if self.health is not None:
            self.health.start_monitor()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['my_mission_mask']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,target=self.gt_image)

                else:
                    if self.task in ['my_mission_mask']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,target=self.gt_image)
  
                if self.health is not None:
                    self.health.monitor_batch(self.netG, self.cond_image, self.gt_image, self.mask, self.loss_mask, self.output)
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        if self.health is not None:
            self.health.end_monitor(self.epoch)
        return self.val_metrics.result()

    def test(self):
        net = self.netG_EMA if self.test_ema else self.netG
        self.logger.info('test: sampling with the {} weights'.format('EMA' if self.test_ema else 'netG'))
        net.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['my_mission_mask']:
                        self.output, self.visuals = net.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,target=self.gt_image)
                else:
                    if self.task in ['my_mission_mask']:
                        self.output, self.visuals = net.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,target=self.gt_image)


                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})
        
        with open("ddim_test.txt", "a") as file:  # 
            file.write(self.logger.opt['path']['resume_state']+"   ")
            file.write(str(test_log['test/mae'].cpu().numpy()) + "\n")
        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))
        
    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
