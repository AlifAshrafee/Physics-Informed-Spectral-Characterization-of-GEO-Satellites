import argparse
from utils.train_vae import train


if __name__ == "__main__":
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

    # model params
    parser.add_argument('--latent-dim', type=int, default=32)

    # training params
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0, help='recon loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='kld loss weight')
    parser.add_argument('--gamma', type=float, default=1.0, help='abundance loss weight')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train(train_spectra=args.train_spectra, train_abundances=args.train_abundances,
          val_spectra=args.val_spectra, val_abundances=args.val_abundances,
          test_spectra=args.test_spectra, test_abundances=args.test_abundances,
          endmembers=args.endmembers, results_dir=args.results_dir, latent_dim=args.latent_dim,
          batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
          alpha=args.alpha, beta=args.beta, gamma=args.gamma,
          patience=args.patience, seed=args.seed)