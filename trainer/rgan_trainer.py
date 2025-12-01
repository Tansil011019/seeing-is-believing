import logging
import torch
from torch.autograd import grad, Variable
from torch.optim import Adam
import os
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)

class RGANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        dataloader, 
        device,
        epoch,
        checkpoint_dir="./rgan_checkpoints",
        checkpoint_interval=1000,
        resume_path=None,
        save_path=None,
        z_dim=100,
        beta=0.5,
        lambda_sim=0.1,
        lambda_gp=10,
        generator_lr=0.0001,
        discriminator_lr=0.0001,
        discriminator_training_times=4,
        generator_training_times=3,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.device = device
        self.z_dim = z_dim
        self.save_path = save_path

        self.optimizer_generator = Adam(self.generator.parameters(), lr=generator_lr, betas=(beta, 0.999))
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(beta, 0.999))

        self.lambda_gp = lambda_gp
        self.lambda_sim = lambda_sim
        self.discriminator_training_times = discriminator_training_times
        self.generator_training_times = generator_training_times
        if epoch is None:
            raise ValueError("You must provide 'epoch' to RGANTrainer!")
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.resume_path = resume_path
        self.checkpoint_interval = checkpoint_interval
        
    def _compute_gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates, _ = self.discriminator(interpolates)

        fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
        
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            logger.error(f"Checkpoint file {file_path} not found.")
            return
        
        checkpoint = torch.load(file_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
        logger.info(f"Loaded checkpoint from {file_path}")

        return checkpoint.get('epoch', 0) + 1
    
    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"rgan_checkpoint_epoch_{epoch}.pth")
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        discriminator_losses = []
        generator_losses = []

        start_epoch = 0
        logger.info(self.resume_path)
        if self.resume_path:
            start_epoch = self.load_checkpoint(self.resume_path)
        
        self.generator.train()
        self.discriminator.train()

        logger.info(f"Starting R-GAN Training for {self.epoch} epochs...")

        for epoch in range(start_epoch, self.epoch):
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Epoch {}/{}".format(epoch+1, self.epoch))

            for i, (imgs, _) in progress_bar:
                imgs = imgs.to(self.device)
                batch_size = imgs.shape[0]

                for _ in range(self.discriminator_training_times):
                    self.optimizer_discriminator.zero_grad()
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_imgs = self.generator(z).detach()

                    real_validity, _ = self.discriminator(imgs)
                    fake_validity, _ = self.discriminator(fake_imgs)
                    gp = self._compute_gradient_penalty(imgs.data, fake_imgs.data)

                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gp
                    d_loss.backward()
                    self.optimizer_discriminator.step()

                for _ in range(self.generator_training_times):
                    self.optimizer_generator.zero_grad()
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    gen_imgs = self.generator(z)
                    fake_validity, _ = self.discriminator(gen_imgs)

                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    self.optimizer_generator.step()

            logger.info(f"Epoch [{epoch+1}/{self.epoch}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            discriminator_losses.append(d_loss.item())
            generator_losses.append(g_loss.item())

            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        history = {
            'discriminator_losses': discriminator_losses,
            'generator_losses': generator_losses
        }

        if self.save_path is not None:
            datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logger.info(f"Saving final model to {self.save_path}")
            final_model_path = os.path.join(self.save_path, f"rgan_final_model_{datastamp}.pth")
            os.makedirs(self.save_path, exist_ok=True)
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
            }, final_model_path)
        
        return history
            