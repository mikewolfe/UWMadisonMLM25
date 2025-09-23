import pandas as pd
SEQ_DICT = {row['ID']:row['Sequence'] for _,row in pd.read_csv('Config/sequence_data.csv').iterrows()}

def LoadSequenceData_FromDfRow(df_row):
    position = int(df_row['pos'])
    alt_short = df_row['alt_short']
    wt_seq_id = df_row['ensp']
    return LoadSequenceData(wt_seq_id, position, alt_short)

def LoadSequenceData(wt_seq_id, position, alt_short):
    seq = SEQ_DICT[wt_seq_id]
    if(alt_short == '*'):
        return seq[:position-1]
    return seq[:position-1] + alt_short + seq[position:]