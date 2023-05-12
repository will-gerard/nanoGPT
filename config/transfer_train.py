import time
wandb_log = True
wandb_project = 'NanoGPT-TransferLearning'
wandb_run_name='transfer-test' + str(time.time())

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8
block_size = 256
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 5
eval_iters = 5
log_interval = 5

# weight decay
weight_decay = 1e-1

# the model does seem to overfit, only save a checkpoint if
# validation loss is lower
always_save_checkpoint = False

pretrained_model_dir = 'pretrained'
model_file_name = #TODO: model file name here
init_from = 'transfer'