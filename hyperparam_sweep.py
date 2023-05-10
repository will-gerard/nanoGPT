import wandb
from contextlib import nullcontext

from eval_transfer import load_transfer_model
from mohler_dataset_preprocessing import load_test_data, get_batch

import torch

import os

## TODO: move some repeated config into a file
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
pretrained_model_dir = 'pretrained'
eval_interval = 30
log_interval = 1
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'transfer' # 'scratch' or 'resume' or 'gpt2*' or 'transfer'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

sweep_configuration = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'validation_loss'},
    'parameters': 
    {
        'learning_rate': {'values': [0.0001, 0.00025, 0.0005, 0.001]},
        'dropout': {0, 0.1, 0.2, 0.3, 0.4, 0.5},
        'regression_network_layers': {'values': [1, 2]},
        'regression_network_hidden_size': {'values': [32, 64, 128]},
        'regression_network_activation': {'values': ['relu', 'tanh']},
        'bias_enabled': {'values': [True, False]},
        'batch_size': {'values': [8, 16, 32]},
        'gradient_accumulation_steps': {'values': [1, 2, 4, 8, 16]},
        'weight_decay': {'values': [0.0, 0.01, 0.1, 0.5]}
     }
}

@torch.no_grad()
def compute_validation_loss(model, ctx, eval_iters=20):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, _, _ = get_batch(split)
            with ctx:
                logits, loss, _, _, _, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main_training_loop(config):
    # do training
    model = load_transfer_model(config)

    model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.learning_rate, (beta1, beta2), 
                device_type)
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # return loss
    return validation_loss

def main():
    with wandb.init(project='hpml-hw3'):
      val_loss = main_training_loop(wandb.config)
      wandb.log({'validation_loss': val_loss})

if __name__ == '__main__':
    # do some setup configuration
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # start the actual sweep
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)
    
    out_dir = f'./sweeps/{sweep_id}'
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    # load the data
    x_train, y_train, x_val, y_val = load_test_data()