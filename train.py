"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, TransferGPT

import torch.nn.functional as F


import pandas as pd
import numpy as np
import tiktoken
import random

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
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from != "transfer":
    # poor man's data loader
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        tensor_form_start = time.perf_counter()
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        tensor_form_time = time.perf_counter() - tensor_form_start
        data_transfer_start = time.perf_counter()
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        data_transfer_time = time.perf_counter() - data_transfer_start
        return x, y, tensor_form_time, data_transfer_time
else:
    # if we we are in transfer mode, we don't want to load the openweb dataset
    # we want to load Mohler instead
    data_dir = os.path.join('data', dataset)
    data = pd.read_csv("mohler_dataset_edited.csv")

    print(f"the size of the full dataset is {data.shape}")

    # Split the DataFrame into training, validation, and testing sets
    train, validate, test = np.split(data.sample(frac=1, random_state=42), [int(.7*len(data)), int(.9*len(data))])

    # Print the sizes of the resulting sets
    print("Training set size: ", len(train))
    print("Validation set size: ", len(validate))
    print("Testing set size: ", len(test))

    # First we want to transform this data into X, Y pairs
    # Each X will be the (question, desired_answer, student_answer)
    # Y will be the corresponding score_avg
    # Define a list of column names to select
    selected_cols = ['question', 'desired_answer', 'student_answer']

    # same process function as in prepare.py for the openweb dataset, I'm guessing we want the format to be the same?
    # this time just take in the text directly
    # and output the encod
    enc = tiktoken.get_encoding("gpt2")
    def process(text):
        ids = enc.encode_ordinary(text) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        return ids

    # process each of the columns we care about in the dataframe
    x_df = train[selected_cols]
    val_df = validate[selected_cols]
    # Apply the process function to each element of the selected columns
    encoded_dataframe = x_df.applymap(process)
    encoded_val_dataframe = val_df.applymap(process)

    # # He has some fancy concatenation thing, writing this all to a file, its a little confusing
    # # our dataset is small, I'm not going to worry about it, and will create tensors directly
    X_tuples = []
    for index, row in encoded_dataframe.iterrows():
        question_tensor = torch.tensor(row['question'], dtype=torch.int64)
        desired_answer_tensor = torch.tensor(row['desired_answer'], dtype=torch.int64)
        student_answer_tensor = torch.tensor(row['student_answer'], dtype=torch.int64)
        X_tuples.append((question_tensor, desired_answer_tensor, student_answer_tensor))
    x_data_joined = []
    for tup in X_tuples:
        x_tensor = torch.cat([tup[0], tup[1], tup[2]])
        x_data_joined.append(x_tensor)
    
    # do same thing with val dataset
    val_tuples = []
    for index, row in encoded_val_dataframe.iterrows():
        question_tensor = torch.tensor(row['question'], dtype=torch.int64)
        desired_answer_tensor = torch.tensor(row['desired_answer'], dtype=torch.int64)
        student_answer_tensor = torch.tensor(row['student_answer'], dtype=torch.int64)
        val_tuples.append((question_tensor, desired_answer_tensor, student_answer_tensor))
    val_data_joined = []
    for tup in val_tuples:
        val_tensor = torch.cat([tup[0], tup[1], tup[2]])
        val_data_joined.append(val_tensor)
    
    # now get the average scores, which will be our y values
    # move these to the GPU directly, the dataset is small so we can afford to keep it in GPU 
    # memory the entire time
    y_train_data = np.array(train['score_avg'])
    y_train_tensor = torch.tensor(y_train_data).to(device)
    y_val_data = np.array(validate['score_avg'])
    y_val_tensor = torch.tensor(y_val_data).to(device)

    # Now we want to perform one extra step, and pad all the x tensors so they are all the same length
    # length should be block size
    padded_train = []
    for sample in x_data_joined:
        if len(sample) > 255: # I think only one or two samples should meet this, allows less padding
            print("WARN: dropping sample from training set, length longer than 255")
            continue
        else:
            pad_length = block_size - len(sample)
            padded_sample = F.pad(sample, (pad_length, 0), mode='constant', value=0)
            padded_train.append(padded_sample)
    
    # convert to tensors and move these tensors to the GPU up front
    padded_train = torch.stack([torch.tensor(sample) for sample in padded_train]).to(device)

    # again, do same operation on val
    padded_val = []
    for sample in val_data_joined:
        if len(sample) > 255: # I think only one or two samples should meet this, allows less padding
            print("WARN: dropping sample from testing set, length longer than 255")
            # TODO: now also need to drop corresponding sample from Y!
            continue
        else:
            pad_length = block_size - len(sample)
            padded_sample = F.pad(sample, (pad_length, 0), mode='constant', value=0)
            padded_val.append(padded_sample)
    
    padded_val = torch.stack([torch.tensor(sample) for sample in padded_val]).to(device)

    print("Done loading and preparing Mohler dataset")

    def get_batch(split):
        tensor_form_start = time.perf_counter()
        dataset = padded_train if split == 'train' else padded_val
        y_dataset = y_train_tensor if split == 'train' else y_val_tensor 
        sampled_indices = random.sample(range(0,len(dataset)), batch_size)
        x_batch = torch.stack([dataset[i] for i in sampled_indices])
        y_batch = torch.stack([y_dataset[i] for i in sampled_indices])
        tensor_form_time = time.perf_counter() - tensor_form_start

        data_transfer_start = time.perf_counter()
        # Now we should already have x and y on the correct device
        # will still call to for now to time and confirm that this is
        # now extremely fast
        x, y = x_batch.to(device), y_batch.to(device)
        data_transfer_time = time.perf_counter() - data_transfer_start
        return x,y, tensor_form_time, data_transfer_time

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'transfer':
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
    print("Created transfer learning model successfully!")
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'mingyu-ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
#scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
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

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, _, _ = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    total_data_loading_time = 0
    data_loading_tensor_form_time = 0
    data_loading_tensor_transfer_time = 0
    total_forward_pass_time = 0
    total_backward_pass_time = 0
    total_validation_set_eval_time = 0
    total_log_data_transfer_time = 0
    total_embed_time = 0
    total_transformer_time = 0
    total_linear_layer_time = 0
    total_loss_compute_time = 0

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        estimate_loss_start_time = time.perf_counter()
        losses = estimate_loss()
        total_validation_set_eval_time += (time.perf_counter() - estimate_loss_start_time)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            forward_pass_start_time = time.perf_counter()
            logits, loss, embed_time, pretrained_forwarding_time, linear_layer_time, loss_compute_time = model(X, Y)
            total_forward_pass_time += (time.perf_counter() - forward_pass_start_time)
            total_embed_time += embed_time
            total_transformer_time += pretrained_forwarding_time
            total_linear_layer_time += linear_layer_time
            total_loss_compute_time += loss_compute_time
            loss = loss.float()
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        data_load_start_time = time.perf_counter()
        X, Y, form_time, transfer_time = get_batch('train')
        total_data_loading_time += (time.perf_counter() - data_load_start_time)
        data_loading_tensor_form_time += form_time
        data_loading_tensor_transfer_time += transfer_time
        # backward pass, with gradient scaling if training in fp16
        backward_pass_start_time = time.perf_counter()
        loss.backward()
        total_backward_pass_time += (time.perf_counter() - backward_pass_start_time)
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
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        log_transfer_start = time.perf_counter()
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        total_log_data_transfer_time += (time.perf_counter() - log_transfer_start)
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        print(f"Printing average perf counter times for the last {log_interval} iterations:")
        print(f"Total forward pass time: {total_forward_pass_time}")
        print(f"Total forward pass embedding time: {total_embed_time}")
        print(f"Total forward pass transformer time: {total_transformer_time}")
        print(f"Total forward pass linear layer time: {total_linear_layer_time}")
        print(f"Total forward pass loss comp time: {total_loss_compute_time}")
        print(f"Average backward pass time: {total_backward_pass_time}")
        print(f"Average data loading time: {total_data_loading_time}")
        print(f"Average tensor form time: {data_loading_tensor_form_time}")
        print(f"Average data transfer time: {data_loading_tensor_transfer_time}")
        print(f"Average validation set eval time: {total_validation_set_eval_time}")
        print(f"Average logging time: {total_log_data_transfer_time}")
        print(f"Printing cuda memory summary from pytorch:")
        print(torch.cuda.memory_summary())
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
