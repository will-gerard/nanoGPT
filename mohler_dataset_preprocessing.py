import pandas as pd
import random
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

## file with some useful functions for preprocessing the mohler dataset
# same process function as in prepare.py for the openweb dataset, I'm guessing we want the format to be the same?
# this time just take in the text directly
# and output the encod
enc = tiktoken.get_encoding("gpt2")
def process(text):
    ids = enc.encode_ordinary(text) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    return ids

def preprocess_dataset(data: pd.DataFrame, device, block_size, encoding_function):
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
    

    # process each of the columns we care about in the dataframe
    x_df = train[selected_cols]
    val_df = validate[selected_cols]
    # Apply the process function to each element of the selected columns
    encoded_dataframe = x_df.applymap(encoding_function)
    encoded_val_dataframe = val_df.applymap(encoding_function)

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

    return padded_train, y_train_tensor, padded_val, y_val_tensor

def get_batch(x, y, batch_size, device):
    sampled_indices = random.sample(range(0,len(x)), batch_size)
    x_batch = torch.stack([x[i] for i in sampled_indices])
    y_batch = torch.stack([y[i] for i in sampled_indices])
    x, y = x_batch.to(device), y_batch.to(device)
    return x_batch, y_batch