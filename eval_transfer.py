import os
import pandas as pd
import numpy as np
import tiktoken
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from model import GPTConfig, GPT, TransferGPT

# set some params for eval
block_size = 256
device = 'cuda'
saved_model_dir = 'transfer_learning_results'
saved_model_name = 'pruned_transfer_output.pt'

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

dtype = 'float32' 

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_test_data():
    data = pd.read_csv("mohler_dataset_edited.csv")

    print(f"the size of the full dataset is {data.shape}")

    # Split the DataFrame into training, validation, and testing sets
    _, _, test = np.split(data.sample(frac=1, random_state=42), [int(.7*len(data)), int(.9*len(data))])

    # Print the sizes of the resulting sets
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
    test_df = test[selected_cols]
    # Apply the process function to each element of the selected columns
    encoded_test_dataframe = test_df.applymap(process)

    test_tuples = []
    for index, row in encoded_test_dataframe.iterrows():
        question_tensor = torch.tensor(row['question'], dtype=torch.int64)
        desired_answer_tensor = torch.tensor(row['desired_answer'], dtype=torch.int64)
        student_answer_tensor = torch.tensor(row['student_answer'], dtype=torch.int64)
        test_tuples.append((question_tensor, desired_answer_tensor, student_answer_tensor))
    test_data_joined = []
    for tup in test_tuples:
        test_tensor = torch.cat([tup[0], tup[1], tup[2]])
        test_data_joined.append(test_tensor)

    # now get the average scores, which will be our y values
    y_test_data = np.array(test['score_avg'])

    padded_test = []
    for sample in test_data_joined:
        if len(sample) > 255: # I think only one or two samples should meet this, allows less padding
            print("WARN: dropping sample from testing set, length longer than 255")
            continue
        else:
            pad_length = block_size - len(sample)
            padded_sample = F.pad(sample, (pad_length, 0), mode='constant', value=0)
            padded_test.append(padded_sample)

    x_test = torch.stack(padded_test)
    y_test = torch.stack([torch.tensor(i) for i in y_test_data])
    print(f"shape of x test tensor is {x_test.shape}")
    print(f"shape of y test tensor is {y_test.shape}")
    x, y = x_test.to(device), y_test.to(device)
    print("Done loading and preparing Mohler dataset for evaluation")
    return x,y

def load_transfer_model():
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    print(f"Loading transfer learning model for eval {saved_model_dir}")

    # First, load the model. This works exactly the same way as in the 'resume' case,
    # Except it is a transfer learning model.
    ckpt_path = os.path.join(saved_model_dir, saved_model_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    # force these config attributes to be equal for proper evaluation of the model
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    # create the model
    gptconf = GPTConfig(**model_args)
    pretrained_model = TransferGPT(pretrained_model=None, config=gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    pretrained_model.load_state_dict(state_dict)
    pretrained_model.eval()
    pretrained_model.to(device)
    print("Created transfer learning model successfully!")

    return pretrained_model

def eval_model():
    print("Starting evaluation!")

    x_test, y_test = load_test_data()
    model = load_transfer_model()
    print("Model and dataset loaded.")

    with torch.no_grad():
        with ctx:
            with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                _, loss = model(x_test, y_test)
                print(f"Got loss of {loss} on the test set.")
            
            print(prof.key_averages().table(row_limit=10))


if __name__ == '__main__':
    eval_model()

