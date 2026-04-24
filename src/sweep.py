from src.train import train

SWEEP_CONFIGS = [
    {
        "run_name": "baseline",
        "learning_rate": 5e-5,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
    },
    {
        "run_name": "conservative",
        "learning_rate": 2e-5,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
    },
    {
        "run_name": "aggressive",
        "learning_rate": 1e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
    },
]

if __name__ == "__main__":
    for i, config in enumerate(SWEEP_CONFIGS, 1):
        print(f"\n── Run {i}/{len(SWEEP_CONFIGS)}: {config['run_name']} ──\n")
        train(**config)
