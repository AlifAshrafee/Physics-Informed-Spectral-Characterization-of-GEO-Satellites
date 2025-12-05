import argparse
import json
from pathlib import Path
from utils.train_vae import train


def load_best_params(best_params_file='results/best_params.json'):
    if Path(best_params_file).exists():
        with open(best_params_file, 'r') as f:
            data = json.load(f)
            best_params = data['best_params']
    else:
        print(f"Warning: optimal hyperparameters file not found! Using default hyperparameters.")
        best_params = {
            'alpha': 1.0,
            'beta': 0.1,
            'gamma': 1.0,
            'delta': 1.0,
            'lr': 1e-3
        }

    return best_params


if __name__ == "__main__":
    best_params = load_best_params('results/best_params.json')

    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument('--train-spectra', default='data/synthetic_dataset_splits/train_spectra.csv')
    parser.add_argument('--train-abundances', default='data/synthetic_dataset_splits/train_abundances.csv')
    parser.add_argument('--val-spectra', default='data/synthetic_dataset_splits/val_spectra.csv')
    parser.add_argument('--val-abundances', default='data/synthetic_dataset_splits/val_abundances.csv')
    parser.add_argument('--test-spectra', default='data/synthetic_dataset_splits/test_spectra.csv')
    parser.add_argument('--test-abundances', default='data/synthetic_dataset_splits/test_abundances.csv')
    parser.add_argument('--endmembers', default='data/endmember_library_clipped.csv')
    parser.add_argument('--results-dir', default='results/vae')

    # training params
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=best_params['lr'])
    parser.add_argument('--alpha', type=float, default=best_params['alpha'], help='recon loss weight')
    parser.add_argument('--beta', type=float, default=best_params['beta'], help='kld loss weight')
    parser.add_argument('--gamma', type=float, default=best_params['gamma'], help='group loss weight')
    parser.add_argument('--delta', type=float, default=best_params['delta'], help='categorical loss weight')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    results = train(train_spectra=args.train_spectra, train_abundances=args.train_abundances,
                    val_spectra=args.val_spectra, val_abundances=args.val_abundances,
                    test_spectra=args.test_spectra, test_abundances=args.test_abundances,
                    endmembers=args.endmembers, batch_size=args.batch_size, 
                    epochs=args.epochs, lr=args.lr,
                    alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
                    patience=args.patience, seed=args.seed, results_dir=args.results_dir)
