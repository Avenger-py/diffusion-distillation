class Config:
    img_size = 64
    batch_size = 32 # may need to reduce to 16 for distillation
    T = 1000
    sample_T = 1000
    device = "cuda"
    lr = 0.001
    s_lr = 0.0001 # lr for student model (distillation)
    time_embd_dim = 32
    steps = 50
    max_iters = steps * (16000 // batch_size) # num iterations through all batches in dataset ( ~ 16000/batch_size)

config = Config()
