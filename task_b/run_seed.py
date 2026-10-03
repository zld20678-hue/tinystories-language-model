import argparse

import train_context_experiment as experiment_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("baseline", "curriculum"), required=True)
    parser.add_argument("--budget", choices=("pilot", "full"), default="full")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # The main runner was originally written around a fixed seed=42.
    # Override that module-level value before starting the run so every
    # stochastic component and every output filename uses the requested seed.
    experiment_runner.SEED = args.seed
    experiment_runner.run(args.experiment, args.budget)


if __name__ == "__main__":
    main()
