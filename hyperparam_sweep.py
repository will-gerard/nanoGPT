import wandb
from contextlib import nullcontext

from eval_transfer import load_transfer_model
from mohler_dataset_preprocessing import preprocess_dataset, get_batch, process
from model import GPT, GPTConfig, TransferGPT

import torch
import pandas as pd

import time

import os

## TODO: move some repeated config into a file
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
pretrained_model_dir = 'pretrained'
pretrained_model_checkpoint_name = 'ckpt_pruned.pt'
eval_interval = 5
log_interval = 5
eval_iters = 5
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'transfer' # 'scratch' or 'resume' or 'gpt2*' or 'transfer'
# wandb logging
wandb_log = True
wandb_project = 'NanoGPT-TransferLearning'
wandb_run_name='transfer-test' + str(time.time())
# data
dataset = 'openwebtext'
block_size = 256
# model
n_layer = 12
n_head = 12
n_embd = 768
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
max_iters = 600000 # total number of training iterations
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

gradient_accumulation_steps = 40

@torch.no_grad()
def compute_validation_loss(model, ctx, x_val, y_val, eval_batch_size, eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(x_val, y_val, eval_batch_size, device)
        with ctx:
            logits, loss, _, _, _, _ = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, 
            sweep_save_directory, checkpoint_name):
    checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
    print(f"saving checkpoint to {sweep_save_directory}")
    torch.save(checkpoint, os.path.join(sweep_save_directory, checkpoint_name))

def initialize_model(dropout):
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    print(f"Starting Transfer Learning from pretrained model saved in {pretrained_model_dir}")

    # First, load the model. This works exactly the same way as in the 'resume' case,
    # Except it is a transfer learning model.
    ckpt_path = os.path.join(pretrained_model_dir, 'ckpt_pruned.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    pretrained_model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    pretrained_model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Once the model is loaded, feed it into the constructor of the new transferGPT class
    model = TransferGPT(pretrained_model=pretrained_model, config=gptconf)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
    print("Created transfer learning model successfully!")

    return model, model_args


def main_training_loop(config, iterations, ctx, x_train, y_train, x_val, y_val, model_config, sweep_save_directory):
    # override global variables for this run
    dropout = config.dropout
    batch_size = config.batch_size
    # gradient_accumulation_steps = config.gradient_accumulation_steps
    weight_decay = config.weight_decay
    learning_rate = config.learning_rate

    checkpoint_name = f'checkpoint_{batch_size}_{dropout}_{gradient_accumulation_steps}_{weight_decay}_{learning_rate}.pt'

    # initialize model with the config we want for this sweep
    model, model_args = initialize_model(dropout)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 
                device_type)
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # do training
    iter = 1 # start at 1 to avoid logging on first iter
    lowest_validation_loss = float('inf')
    X, Y  = get_batch(x_train, y_train, batch_size, device) # fetch the very first batch
    t0 = time.time()
    while iter < iterations:
        # train for one iteration
            # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss, embed_time, pretrained_forwarding_time, linear_layer_time, loss_compute_time = model(X, Y)
                loss = loss.float()
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(x_train, y_train, batch_size, device)
            # backward pass, with gradient scaling if training in fp16
            loss.backward()
        # clip the gradient
        if grad_clip != 0.0:
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        optimizer.step()
        # scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            #print(f"iter {iter}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            if wandb_log:
                wandb.log({'train_loss': lossf})
        # validate
        if iter % eval_interval == 0:
            validation_loss = compute_validation_loss(model, ctx, x_val, y_val, batch_size, eval_iters)
            if wandb_log:
                wandb.log({'validation_loss': validation_loss})
            # save checkpoint
            if validation_loss < lowest_validation_loss:
                save_checkpoint(model, optimizer, model_args, iter, validation_loss,
                    model_config, sweep_save_directory, checkpoint_name)
                lowest_validation_loss = validation_loss
        iter += 1

    # return loss
    return lowest_validation_loss 

def main(config, sweep_save_directory):
    with wandb.init(project='hpml-hw3'):
      best_val_loss = main_training_loop(wandb.config, 200, ctx, x_train, y_train, x_val, y_val, config, sweep_save_directory)
      wandb.log({'best_validation_loss': best_val_loss})

if __name__ == '__main__':
    # do some setup configuration
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    # tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    # print(f"tokens per iteration will be: {tokens_per_iter:,}")
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # start the actual sweep
    wandb.login()
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'best_validation_loss'},
        'parameters': 
        {
            'learning_rate': {'values': [0.00005, 0.0001, 0.0005, 0.001]},
            'dropout': {'values': [0, 0.1, 0.5]},
            'batch_size': {'values': [8, 32, 128]},
            # 'gradient_accumulation_steps': {'values': [5, 10, 25]},
            'weight_decay': {'values': [0.0, 0.01, 0.1, 0.5]},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)
    
    out_dir = f'./sweeps/{sweep_id}'
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    # load the data
    data = pd.read_csv("mohler_dataset_edited.csv")
    x_train, y_train, x_val, y_val = preprocess_dataset(data, device, block_size, process)

    # run the sweep
    wandb.agent(sweep_id, lambda: main(config, out_dir), count=10)