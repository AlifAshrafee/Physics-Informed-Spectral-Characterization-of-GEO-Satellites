import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from utils.train_vae import train


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.5, 5.0, log=True)
    beta = trial.suggest_float('beta', 0.01, 1.0, log=True)
    gamma = trial.suggest_float('gamma', 1.0, 20.0, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64])

    print(f"Trial {trial.number + 1}")
    print(f"alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")

    try:
        results = train(
            train_spectra='data/synthetic_dataset_splits/train_spectra.csv',
            train_abundances='data/synthetic_dataset_splits/train_abundances.csv',
            val_spectra='data/synthetic_dataset_splits/val_spectra.csv',
            val_abundances='data/synthetic_dataset_splits/val_abundances.csv',
            test_spectra='data/synthetic_dataset_splits/test_spectra.csv',
            test_abundances='data/synthetic_dataset_splits/test_abundances.csv',
            endmembers='data/endmember_library_clipped.csv',
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lr=lr,
            latent_dim=latent_dim,
            epochs=100,
            patience=10,
            batch_size=32,
            seed=42
            )

        trial.report(results['abundance_mae'], step=1)

        trial.set_user_attr('abundance_mae', results['abundance_mae'])
        trial.set_user_attr('abundance_rmse', results['abundance_rmse'])
        trial.set_user_attr('spectral_mse', results['spectral_mse'])
        trial.set_user_attr('spectral_rmse', results['spectral_rmse'])
        trial.set_user_attr('sam_mean', results['sam_mean'])
        trial.set_user_attr('sam_std', results['sam_std'])

        print(f"\nResults:")
        print(f"Abundance MAE: {results['abundance_mae']:.4f}")
        print(f"Abundance RMSE: {results['abundance_rmse']:.4f}")
        print(f"Spectral MSE: {results['spectral_mse']:.4f}")
        print(f"Spectral RMSE: {results['spectral_rmse']:.4f}")
        print(f"SAM Mean: {results['sam_mean']:.4f}°")
        print(f"SAM Std: {results['sam_std']:.4f}°")

        return results['abundance_mae']

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


def run_optimization(n_trials=1000, timeout=None):
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name='vae_mae_optimization'
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Abundance MAE: {best_trial.value:.4f}°")
    print(f"\nAll Metrics for Best Trial:")
    print(f"Abundance MAE: {best_trial.user_attrs['abundance_mae']:.4f}")
    print(f"Abundance RMSE: {best_trial.user_attrs['abundance_rmse']:.4f}")
    print(f"Spectral MSE: {best_trial.user_attrs['spectral_mse']:.4f}")
    print(f"Spectral RMSE: {best_trial.user_attrs['spectral_rmse']:.4f}")
    print(f"SAM Mean: {best_trial.user_attrs['sam_mean']:.4f}°")
    print(f"SAM Std: {best_trial.user_attrs['sam_std']:.4f}°")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value:.4f}")

    results = {
        'optimization_metric': 'abundance_mae',
        'best_params': best_trial.params,
        'best_abundance_mae': best_trial.value,
        'best_metrics': {
            'abundance_mae': best_trial.user_attrs['abundance_mae'],
            'abundance_rmse': best_trial.user_attrs['abundance_rmse'],
            'spectral_mse': best_trial.user_attrs['spectral_mse'],
            'spectral_rmse': best_trial.user_attrs['spectral_rmse'],
            'sam_mean': best_trial.user_attrs['sam_mean'],
            'sam_std': best_trial.user_attrs['sam_std'],
        },
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'abundance_mae': t.user_attrs.get('abundance_mae'),
                'abundance_rmse': t.user_attrs.get('abundance_rmse'),
                'spectral_mse': t.user_attrs.get('spectral_mse'),
                'spectral_rmse': t.user_attrs.get('spectral_rmse'),
                'sam_mean': t.user_attrs.get('sam_mean'),
                'sam_std': t.user_attrs.get('sam_std'),
            }
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
    }

    with open('results/best_params.json', 'w') as f:
        json.dump(results, f, indent=2)

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=1000, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    args = parser.parse_args()

    study = run_optimization(n_trials=args.n_trials, timeout=args.timeout)
