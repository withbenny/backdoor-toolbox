import subprocess
import os
import time

def phase1(dataset):
    subprocess.run(
        [
            "python",
            "create_clean_set.py",
            "-dataset",
            dataset,
        ],
        check=True,
    )
    print(f"Phase 1: Create {dataset} clean dataset successfully.")


def phase2(dataset, data_rate, poison_type, poison_rate, cover_rate=None):
    cmd = [
        "python",
        "create_poisoned_set.py",
        "-dataset",
        dataset,
        "-poison_type",
        poison_type,
        "-poison_rate",
        str(poison_rate),
        "-data_rate",
        str(data_rate)
    ]
    if cover_rate is not None:
        cmd += ["-cover_rate", str(cover_rate)]
    subprocess.run(cmd, check=True)
    if poison_type != "none":
        print(f"Phase 2: Create {data_rate} {dataset} poisoned dataset with {poison_type} successfully.")
    else:
        print(f"Phase 2: Create {data_rate} {dataset} clean dataset successfully.")

def phase3(dataset, data_rate, poison_type, poison_rate, num_models, cover_rate=None):
    for i in range(num_models):
        seed = int(time.strftime("%m%d%H%M"))
        cmd = [
            "python",
            "train_on_poisoned_set.py",
            "-dataset",
            dataset,
            "-poison_type",
            poison_type,
            "-poison_rate",
            str(poison_rate),
            "-seed",
            str(seed),
            "-log",
            "-data_rate",
            str(data_rate),
        ]
        if cover_rate is not None:
            cmd += ["-cover_rate", str(cover_rate)]
        subprocess.run(cmd, check=True)
        if poison_type != "none":
            print(f"Phase 3: Train model {i+1}/{num_models} on {data_rate} {dataset} poisoned dataset with {poison_type} successfully.")
        else:
            print(f"Phase 3: Train model {i+1}/{num_models} on {data_rate} {dataset} clean dataset successfully.")

def run_process(cfg, poison_type=None):
    dataset = cfg["dataset"]
    if poison_type is None:
        poison_type = cfg["poison_type"]
    data_rate = cfg["data_rate"]
    poison_rate = cfg["poison_rate"]
    cover_rate = None
    num_models = cfg["num_models"]

    # Phase 1: Create Clean Dataset
    # phase1(dataset)

    # Phase 2: Create Poisoned Dataset
    if poison_type in ["WaNet", "TaCT", "adaptive_blend", "adaptive_patch"]:
        cover_rate = cfg[poison_type]["cover_rate"]
    phase2(dataset, data_rate, poison_type, poison_rate, cover_rate)

    # Phase 3: Train Models
    phase3(dataset, data_rate, poison_type, poison_rate, num_models, cover_rate)

if __name__ == "__main__":
    import tomllib
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)
    # poison_types = [
    #     "badnet", "blend", "trojan", "SIG", "dynamic", "ISSBA",
    #     "WaNet", "TaCT", "adaptive_blend", "adaptive_patch"
    # ]
    poison_types = ["blend"]
    for poison_type in poison_types:
        print(f"=== Running for poison_type: {poison_type} ===")
        run_process(cfg["BDTrainer"], poison_type)