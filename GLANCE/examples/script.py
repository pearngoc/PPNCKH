import sys
import traceback
import copy
from typing import Literal, Optional, List
from pprint import pprint
from argparse import ArgumentParser
import time
import json
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from raiutils.exceptions import UserErrorException
from pathlib import Path

from glance.glance.glance import GLANCE
from glance.local_cfs.dice_method import DiceMethod
from glance.local_cfs.nearest_neighbor import NearestNeighborMethod, NearestNeighborsScaled
from glance.local_cfs.random_sampling import RandomSampling
from glance.utils.metadata_requests import _decide_local_cf_method
from glance.counterfactual_costs import build_dist_func_dataframe

from utils import load_models, preprocess_datasets, preprocess_datasets_kfold


# cost effectiveness size runtime


def run_experiment(
    dataset: str,
    model: str,
    method: Literal["GLANCE", "Local"],
    local_cf_generator: Literal[
        "Dice", "NearestNeighbors", "NearestNeighborsScaled", "RandomSampling"
    ],
    clustering_method: Literal["KMeans","Agglomerative",'GMM'],
    n_initial_clusters: int,
    n_final_clusters: int,
    n_local_counterfactuals: int,
    IM__cluster_action_choice_algo: Literal["max-eff", "mean-act", "low-cost"],
    IM__nns__n_scalars: int = 5,
    IM__rs__n_most_important: Optional[int] = None,
    IM__rs__n_categorical_most_frequent: Optional[int] = None,
    IM__lowcost__action_threshold: Optional[int] = None,
    IM__lowcost__num_low_cost: Optional[int] = None,
    IM__min_cost_eff_thres__effectiveness_threshold: Optional[float] = None,
    IM__min_cost_eff_thres_combinations__num_min_cost: Optional[int] = None,
    ΙΜ__eff_thres_hybrid__max_n_actions_full_combinations: Optional[int] = None,
    IM__save_single_action_metrics: Optional[bool] = None,
):

    data, affected_list, _unaff, model_list, train_dataset_list, feat_to_vary, target_name, _numfeats, _catfeats = (
        preprocess_datasets_kfold(dataset, load_models(dataset, model), model)
    )
    print("done with preprocess")
    effs = []
    costs = []
    times = []
    final_sizes = []

    start_time = time.time()
    if method == "GLANCE":
        if local_cf_generator == "NearestNeighborsScaled" and np.isnan(
            IM__nns__n_scalars
        ):
            raise ValueError(
                "Must provide number of scalars for Nearest Neighbors Scaled"
            )
        else:
            IM__nns__n_scalars = int(IM__nns__n_scalars)
        for i in range(len(affected_list)):
            start_time = time.time()
            affected = affected_list[i]
            train_dataset = train_dataset_list[i]
            model_ = model_list[i]

            global_method = GLANCE(
                model_,
                initial_clusters=n_initial_clusters,
                final_clusters=n_final_clusters,
                num_local_counterfactuals=n_local_counterfactuals,
            )

            global_method.fit(
                data.drop(columns=[target_name]),
                data[target_name],
                train_dataset,
                feat_to_vary,
                cf_generator=local_cf_generator,
                clustering_method=clustering_method,
                cluster_action_choice_algo=IM__cluster_action_choice_algo,
                nns__n_scalars=IM__nns__n_scalars,
                rs__n_most_important=IM__rs__n_most_important,
                rs__n_categorical_most_frequent=IM__rs__n_categorical_most_frequent,
                lowcost__action_threshold=IM__lowcost__action_threshold,
                lowcost__num_low_cost=IM__lowcost__num_low_cost,
                min_cost_eff_thres__effectiveness_threshold=IM__min_cost_eff_thres__effectiveness_threshold,
                min_cost_eff_thres_combinations__num_min_cost=IM__min_cost_eff_thres_combinations__num_min_cost,
                eff_thres_hybrid__max_n_actions_full_combinations=ΙΜ__eff_thres_hybrid__max_n_actions_full_combinations,
            )
            n_flipped, cost, clusters, clusters_res, chosen_actions, final_costs = global_method.explain_group(affected)
            eff = n_flipped / affected.shape[0]
            mean_cost = cost / n_flipped
            n_global_cfs = n_final_clusters
            
            end_time = time.time()
            total_time = end_time - start_time

            if IM__save_single_action_metrics:
                single_action_metrics = global_method.single_action_metrics()
                cache_dir = (
                    Path(".") / "cache" / "single_action_costs" / 
                    dataset / model / f"{n_initial_clusters=}" / f"{n_final_clusters=}" / f"{n_local_counterfactuals=}"
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                single_action_metrics.to_csv(cache_dir / f"single_action_metrics_fold_{i}.csv", index=False)
            
            effs.append(eff)
            costs.append(mean_cost)
            final_sizes.append(n_global_cfs)
            times.append(total_time)
    elif method == "Local":
        local_cfs_lst = []
        for i in range(len(affected_list)):
            start_time = time.time()
            affected = affected_list[i]
            train_dataset = train_dataset_list[i]
            model_ = model_list[i]

            cf_generator = _decide_local_cf_method(
                method="Dice",
                model=model_,
                train_dataset=train_dataset,
                numeric_features_names=_numfeats,
                categorical_features_names=_catfeats,
                feat_to_vary=feat_to_vary,
                random_seed=13,
            )
            cfs = cf_generator.explain_instances(affected, num_counterfactuals=1)

            dist_func_dataframe = build_dist_func_dataframe(
                X=data.drop(columns=[target_name]),
                numerical_columns=_numfeats,
                categorical_columns=_catfeats,
            )

            aff_with_cf = affected.loc[cfs.index]
            aff_without_cf = affected.drop(cfs.index)

            eff = aff_with_cf.shape[0] / affected.shape[0]
            mean_cost = dist_func_dataframe(aff_with_cf, cfs).mean()

            end_time = time.time()
            total_time = end_time - start_time

            effs.append(eff)
            costs.append(mean_cost)
            final_sizes.append(aff_with_cf.shape[0])
            times.append(total_time)
            local_cfs_lst.append(cfs)
            
        for i, cfs in enumerate(local_cfs_lst):
            directory = Path(f"cache") / "local_cfs" / dataset / model
            directory.mkdir(parents=True, exist_ok=True)
            cfs.to_csv(directory / f"fold_{i}.csv")

    eff = f"{round(100*np.mean(effs), 2)} ± {round(100*np.std(effs), 2)}"
    mean_cost = f"{round(np.mean(costs), 2)} ± {round(np.std(costs), 2)}"
    total_time = f"{round(np.mean(times), 2)} ± {round(np.std(times), 2)}"
    size = f"{round(np.mean(final_sizes), 2)} ± {round(np.std(final_sizes), 2)}"
    return eff, mean_cost, size, total_time


def main():
    parser = ArgumentParser(description="Experiment Runner.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Specify the filename to read param grid from.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Specify file name to write results to.",
    )
    parser.add_argument(
        "-e",
        "--error",
        type=str,
        required=False,
        help="File on which to report errors. Default: stderr",
    )
    parser.add_argument(
        "--append", action="store_true", help="Append to file instead of erasing."
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="If on, stops on error. Else, gathers errors and outputs them at the end.",
    )
    args = parser.parse_args()

    with open(args.input, "r") as inf:
        input_specs = json.load(inf)
    param_grid = ParameterGrid(input_specs["param_grid"])
    exclude_combinations = input_specs["exclude_combinations"]

    def dict_subset(a, b):
        return all(item in b.items() for item in a.items())

    signature = inspect.signature(run_experiment)
    run_experiment_param_names = list(signature.parameters.keys())
    input_columns = run_experiment_param_names
    output_columns = run_experiment_param_names + [
        "Effectiveness",
        "Cost",
        "Size",
        "Elapsed Time",
    ]

    results = pd.DataFrame(columns=output_columns)
    if args.append:
        results = pd.read_csv(args.output)
        pd.testing.assert_index_equal(results.columns, pd.Index(output_columns))
    log = []
    for params in param_grid:
        pprint(params)
        if any(
            dict_subset(exclude_params, params)
            for exclude_params in exclude_combinations
        ):
            continue
        assert all(p_name in run_experiment_param_names for p_name in params.keys())
        if args.stop_on_error:
            eff, cost, size, elapsed = run_experiment(**copy.deepcopy(params))
        else:
            try:
                eff, cost, size, elapsed = run_experiment(**copy.deepcopy(params))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                log.append((params, e))
                continue

        result = pd.DataFrame(columns=output_columns)
        result.loc[0] = params
        result.loc[0, ["Effectiveness", "Cost", "Size", "Elapsed Time"]] = [
            eff,
            cost,
            size,
            elapsed,
        ]
        results = pd.concat([results, result], ignore_index=True)
        results.to_csv(args.output, index=False)


    if args.error is not None:
        with open(args.error, "w") as f:
            print(
                "Failed to run experiments with the following parameters and respective errors:",
                file=f,
            )
            for params, err in log:
                print(params, file=f)
                print(err, file=f)
    else:
        print(
            "Failed to run experiments with the following parameters and respective errors:",
            file=sys.stderr,
        )
        for params, err in log:
            print(params, file=sys.stderr)
            print(err, file=sys.stderr)


if __name__ == "__main__":
    main()
