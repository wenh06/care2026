"""Plan and preprocess with custom planners.

Usage::

    # Default planner
    python scripts/nnunet_custom_plan.py -d 521

    # 4-stage planner (ablation B2)
    python scripts/nnunet_custom_plan.py -d 521 --planner 4stage

    # Verify + plan only
    python scripts/nnunet_custom_plan.py -d 521 --verify --no-pp
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset, preprocess_dataset


class Planner4Stage(ExperimentPlanner):
    """Planner generating 4-stage PlainConvUNet (ablation B2).

    Calls the default planner, then trims all 3d_fullres configurations
    to 4 encoder stages (features [32,64,128,256]), matching VNet depth.
    """

    def plan_experiment(self):
        super().plan_experiment()
        for cfg_name in list(self.plans_manager.plans.get("configurations", {})):
            cfg = self.plans_manager.plans["configurations"][cfg_name]
            arch = cfg.get("architecture", {}).get("arch_kwargs", {})
            n_orig = arch.get("n_stages", 0)
            if n_orig <= 4 or "3d" not in cfg_name:
                continue
            n = 4
            arch["n_stages"] = n
            arch["features_per_stage"] = [32, 64, 128, 256]
            arch["kernel_sizes"] = [[3, 3, 3]] * n
            arch["strides"] = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
            arch["n_conv_per_stage"] = [2, 2, 2, 2]
            arch["n_conv_per_stage_decoder"] = [2, 2, 2]


def main():
    parser = argparse.ArgumentParser(description="Plan and preprocess with custom planners")
    parser.add_argument("-d", type=int, required=True, help="Dataset ID (e.g. 521)")
    parser.add_argument("--planner", type=str, default="default", choices=["default", "4stage"], help="Planner variant")
    parser.add_argument("-c", type=str, default="3d_fullres", help="Configurations (comma-separated)")
    parser.add_argument("--no-pp", action="store_true", help="Skip preprocessing")
    parser.add_argument("--verify", action="store_true", help="Verify dataset integrity before plan")
    parser.add_argument(
        "--overwrite-target-spacing",
        type=str,
        default=None,
        help="Override target spacing, e.g. '2.5,0.625,0.625'",
    )
    args = parser.parse_args()

    dataset_id = args.d
    configs = [c.strip() for c in args.c.split(",")]

    if args.verify:
        from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import verify_dataset_integrity_entry

        verify_dataset_integrity_entry(dataset_id=dataset_id)

    planner_cls = Planner4Stage if args.planner == "4stage" else ExperimentPlanner

    spacing = None
    if args.overwrite_target_spacing:
        spacing = tuple(float(x) for x in args.overwrite_target_spacing.split(","))

    print(f"Planning dataset {dataset_id} with {planner_cls.__name__}...")
    plan_experiment_dataset(dataset_id, planner_cls, overwrite_target_spacing=spacing)

    if not args.no_pp:
        print(f"Preprocessing dataset {dataset_id} ({configs})...")
        preprocess_dataset(dataset_id, "nnUNetPlans", configs)

    print("Done.")


if __name__ == "__main__":
    main()
