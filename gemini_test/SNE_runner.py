import random
import argparse
import numpy as np
import LoadData as data
from SNE import SNE
import json

# Set random seeds
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SNE.")
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path')
    parser.add_argument('--id_dim', type=int, default=20,
                        help='Dimension for id_part.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--n_neg_samples', type=int, default=10,
                        help='Number of negative samples.')
    parser.add_argument('--attr_dim', type=int, default=20,
                        help='Dimension for attr_part.')
    return parser.parse_args()

#################### Util functions ####################


def run_SNE( data, id_dim, attr_dim ):
    model = SNE( data, id_embedding_size=id_dim, attr_embedding_size=attr_dim)
    model.train()
    json_string = json.dumps(model.node_embeddings.tolist())
    json_file = open("model.json", "w")
    json_file.write(json_string)
    json_file.close()
    return model


def main():
    args = parse_args()
    # args = {'data_path': 'week1_data/',
    #     'id_dim': 20,
    #     'attr_dim': 20,
    #     'n_neg_samples': 10,
    #     'epoch': 20}
    print("data_path: ", args.data_path)
    path = args.data_path
    Data = data.LoadData( path , SEED)
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)

    model = run_SNE( Data, args.id_dim, args.attr_dim)
    return locals()


if __name__ == "__main__":
    locals().update(main())