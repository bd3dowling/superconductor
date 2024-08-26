from design_bench.registration import registry, register, make, spec

register(
    "Superconductor-RandomForest-v0",
    "design_bench.datasets.continuous.superconductor_dataset:SuperconductorDataset",
    "design_bench.oracles.sklearn:RandomForestOracle",
    # keyword arguments for building the dataset
    dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=80, min_percentile=0),
    # keyword arguments for building RandomForest oracle
    oracle_kwargs=dict(
        noise_std=0.0,
        max_samples=2000,
        distribution=None,
        max_percentile=100,
        min_percentile=0,
        # parameters used for building the model
        model_kwargs=dict(n_estimators=100, max_depth=100, max_features="auto"),
        # parameters used for building the validation set
        split_kwargs=dict(
            val_fraction=0.5,
            subset=None,
            shard_size=5000,
            to_disk=True,
            disk_target="superconductor/split",
            is_absolute=False,
        ),
    ),
)
