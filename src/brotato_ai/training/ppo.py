"""Active PPO entrypoint facade.

The temporal model implementation remains import-compatible at ``v4`` while
the launcher and runtime ownership move to this package.
"""

from v4.train_temporal_hierarchical import main


if __name__ == "__main__":
    raise SystemExit(main())

