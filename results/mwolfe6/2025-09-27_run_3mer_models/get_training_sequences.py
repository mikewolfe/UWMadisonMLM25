import sys
import pandas as pd
import SequenceFormatter as sf

if __name__ == "__main__":
    train_data = pd.read_csv("data/mave-db-amino-acid-substitution-prediction/train.csv")
    sys.stdout.write("\t".join(train_data.columns) + "\t" + "seq" + "\n")
    for index, row in train_data.iterrows():
        row_to_write = [str(item) for item in row]
        this_seq = sf.LoadSequenceData_FromDfRow(row)
        sys.stdout.write('\t'.join(row_to_write) + '\t' + this_seq + '\n')

