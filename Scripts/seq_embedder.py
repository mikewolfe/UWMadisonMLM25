# This is intended to be run from project directory, requires the following inpuyts 
from transformers import AutoTokenizer, EsmModel
import pandas as pd
import torch
import numpy as np
from LocalLibrary import SequenceFormatter as sf
import argparse
import os
arg_parser = argparse.ArgumentParser(description="Embed Sequences via ESM2")
argparse.add_argument('model_dir', type=str)
argparse.add_argument('data_file', type=str, help='path to data file of variants')
argparse.add_argument('out_dir', type=str, help='output directory')
argparse.add_argument('--batch_size', type=int, help='number of variants to embed at once. lower if OOM error',
                      default=20)

args = arg_parser.parse_args()

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Load models 
tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
model = EsmModel.from_pretrained(os.path.join(args.model_dir, "model"))

# Load data
vnt_table = pd.read_csv(args.data_file)
vnt_table['sequence'] = [
    sf.LoadSequenceData_FromDfRow(vnt_table.iloc[idx,:]) for idx in range(vnt_table.shape[0])
    ]
base_seq_map = sf.SEQ_DICT

vnt_proteins = vnt_table['sequence'].to_list()
tokenized_vnts = tokenizer(vnt_proteins, padding=True, return_tensors='pt')
orig_prots = vnt_table['ensp'].map(base_seq_map).to_list()
tokenized_orig = tokenizer(orig_prots, padding=True, return_tensors='pt')

# Make some batches (THis is for running on a laptop)
batch_size = args.batch_size
inds = [batch_size*i for i in range(tokenized_vnts['input_ids'].shape[0]//batch_size + 1)]
batched_pos = [vnt_table.loc[ind1:ind2-1, 'pos'].to_list() for ind1, ind2 in zip(inds[:-1], inds[1:])]
batched_vnts = [{key:tokenized_vnts[key][ind1:ind2, :] for key in tokenized_vnts.keys()} for ind1, ind2 in zip(inds[:-1], inds[1:])]
batched_orig = [{key:tokenized_orig[key][ind1:ind2, :] for key in tokenized_orig.keys()} for ind1, ind2 in zip(inds[:-1], inds[1:])]


# IK there is a more efficient way to do this too. I should have cached the whole AA level
#       seq embedding and then pulled aas from that cache. This was easier.
model.eval()
batched_vnt_emb = []
batched_orig_emb = []
batched_mut_aa_emb = []
batched_orig_aa_emb = []

for i, batch in enumerate(batched_vnts):
    with torch.no_grad():
        model_out_var = model(**batch)
        model_out_orig = model(**batched_orig[i])
    # Stack the mutants embeddings together
    # last_hidden_state has dimensionality (n_seqs, max_seq_len*, emb_dim)
    #   *I use this loosely theres some padding and stuff
    # BC the first index is padded we can just use the positions who's indices start at 1 so thats nice
    batch_mut_aa_emb = []
    batch_orig_aa_emb = []
    for seq_ind, pos in enumerate(batched_pos[i]):
        batch_mut_aa_emb.append(model_out_var.last_hidden_state[seq_ind, pos, :])
        batch_orig_aa_emb.append(model_out_orig.last_hidden_state[seq_ind, pos, :])
    batched_mut_aa_emb.append(torch.stack(batch_mut_aa_emb, dim=0))
    batched_orig_aa_emb.append(torch.stack(batch_orig_aa_emb, dim=0))
    # pooler output is (n_seqs, emb_dim) These are sequence embeddings
    batched_vnt_emb.append(model_out_var.pooler_output)
    batched_orig_emb.append(model_out_orig.pooler_output)
    print(f'finished batch {i}')
    del model_out_var
    del model_out_orig

np_vnt_embs = torch.cat(batched_vnt_emb, dim=0).detach().numpy()
np.savetxt(os.path.join(out_dir, "var_embs.csv"), np_vnt_embs,
           delimiter=',')
np_orig_embs = torch.cat(batched_orig_emb, dim=0).detach().numpy()
np.savetxt(os.path.join(out_dir, "orig_embs.csv"), np_orig_embs,
           delimiter=',')
np_vnt_aa_embs = torch.cat(batched_mut_aa_emb, dim=0).detach().numpy()
np.savetxt(os.path.join(out_dir, "var_aa_embs.csv"), np_vnt_aa_embs,
           delimiter=',')
np_orig_aa_embs = torch.cat(batched_orig_aa_emb, dim=0).detach().numpy()
np.savetxt(os.path.join(out_dir, "orig_aa_embs.csv"), np_orig_aa_embs,
           delimiter=',')