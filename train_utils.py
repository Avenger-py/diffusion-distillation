import torch
from tqdm import tqdm
from model import Unet
from diffusion import Sampler, distill_loss
from config import config
from diffusion import *

def get_batch(dataloader):
    return next(iter(dataloader))

def save_checkpoint(path, model, optimizer, itr, loss):
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "optimizer": optimizer.state_dict(),
        "epoch": itr,
        "loss": loss,
    }, path)

    print(f"Model saved to {path}")

def load_model(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=config.device)
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
    print(f"\nCheckpoint loaded from {path}\n")
    return model, optimizer

def init_student_model(teacher_model, lr):
    model = Unet(config.time_embd_dim)
    model.to(config.device)
    model.load_state_dict(teacher_model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer
    

def train_model(model, diffusion, iters, batch_size, optimizer, dataloader, T):
    losses = []
    sampler = Sampler(sample_timesteps=T)
    for itr in tqdm(range(iters)):
        optimizer.zero_grad()
        t = torch.randint(0, T, (batch_size,), device=config.device).long()
        x_0 = get_batch(dataloader)[0]
        loss = diffusion.get_loss(model, x_0, t)
        # print(loss)
        loss.backward()
        optimizer.step()

        if (itr % (iters // 4) == 0 or itr == iters - 1) and itr > 0:
            path = f"DDIM_unet_v1_ep{itr}.pt"
            save_checkpoint(path, model, optimizer, itr, loss.item())
        
        if itr % (iters // 50)  == 0:
            print(f"Epoch {itr} | Loss: {loss.item()} ")
            losses.append(loss.item())
        
        if (itr % (iters // 20) == 0) or itr == iters - 1 :
#             diffusion.vis_forward_diffusion(x_0)
            sampler.sample_plot_image(model, T-1)
            
    return model, losses


def train_student(iters, sampling_steps, teacher_model, batch_size, dataloader, T, student_ckpt_path=None):
        teacher_T = T
        
        if student_ckpt_path:
            m = Unet(time_embd_dim).to(config.device)
            student_model, optimizer = load_model(student_ckpt_path, m, torch.optim.AdamW(m.parameters(), lr=config.s_lr))
            print(f"Resuming from student ckpt --> {student_ckpt_path}")
        
        else:
            print(f"Initializing from teacher model......")
            student_model, optimizer = init_student_model(teacher_model, config.s_lr)
            student_model.to(config.device)
        
        #teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        student_model.train()
        teacher_diffusion = Diffusion(timesteps=teacher_T)
        teacher_sampler = Sampler(config.T)
        student_sampler = Sampler(sample_timesteps=teacher_diffusion.T)
        losses = []
        print(f"Training student for {iters} epochs, student sampling steps = {sampling_steps}")
    
        for itr in tqdm(range(iters)):
            optimizer.zero_grad()
            
            # Teacher: T --> T1 --> T2 --> prediction (2 step DDIM)
            # Student: T --> prediction (1 step DDIM)
            f = T // sampling_steps
            idx = torch.randint(1, sampling_steps, (batch_size,), device=config.device).long()
            t = f * idx
            t1 = f * (idx - 0.5)
            t1 = t1.long()
            t2 = f * (idx - 1)
            
            x_0 = get_batch(dataloader)[0]
            loss = distill_loss(x_0, [t, t1, t2], teacher_model.to(config.device), student_model, teacher_diffusion)
            teacher_model.to("cpu")
            loss.backward()
            optimizer.step()
            
            if (itr % (iters // 5) == 0 or itr == iters - 1) and itr > 0 :
                path = f"DDIM_unet_v1_st{sampling_steps}_ep{itr}.pt"
                save_checkpoint(path, student_model, optimizer, itr, loss.item())

            if itr % (iters // 40) == 0:
                print(f"Epoch {itr} | Loss: {loss.item()} ")
                losses.append(loss.item())
            
            if (itr % (iters // 20) == 0) or itr == iters - 1 :
                s_img = torch.randn((1, 3, config.img_size, config.img_size), device=config.device)
                print("Teacher 4x sampling ->")
                teacher_sampler.sample_plot_image(teacher_model.to(config.device), sampling_steps=sampling_steps*4, img=s_img)
                print("Teacher 2x sampling ->")
                teacher_sampler.sample_plot_image(teacher_model.to(config.device), sampling_steps=sampling_steps*2, img=s_img)
                print("Student 1x sampling->")
                student_sampler.sample_plot_image(student_model, sampling_steps=sampling_steps, img=s_img)

        return student_model, losses