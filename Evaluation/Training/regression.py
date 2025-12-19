from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="dataset name", metavar="DATASET", required=True)
args = parser.parse_args()

from Config.config import CONFIG
CONFIG = CONFIG(args.dataset)

from DyGLib.train_link_regression import train
from DyGLib.utils.DataLoader import get_link_prediction_data

if __name__ == '__main__':
    data = get_link_prediction_data(val_ratio=CONFIG.train.val_ratio, 
                            test_ratio=CONFIG.train.test_ratio, 
                            node_dim=CONFIG.model.node_dim)

    result = train(CONFIG.model, CONFIG.data, CONFIG.train, *data)