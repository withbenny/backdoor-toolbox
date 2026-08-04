import subprocess
import os
import time
import json

try:
    import tomllib
except ImportError:
    import tomli as tomllib


OTHER_ATTACK_POISON_TYPES = {"bpp"}

def _has_image_files(root_dir, suffixes):
    if not os.path.isdir(root_dir):
        return False
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in suffixes:
                return True
    return False


def _clean_set_meta_matches(dataset, clean_budget):
    meta_path = os.path.join("clean_set", dataset, "meta.json")
    if not os.path.exists(meta_path):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    return meta.get("clean_budget") == clean_budget


def phase1(dataset, clean_budget):
    clean_set_dir = f"./clean_set/{dataset}"
    image_suffix = {".png", ".jpg", ".jpeg"}
    print(f"=== Running Phase 1 ===")
    if not _has_image_files(clean_set_dir, image_suffix) or not _clean_set_meta_matches(
        dataset, clean_budget
    ):
        subprocess.run(
            [
                "python",
                "create_clean_set.py",
                "-dataset",
                dataset,
                "-clean_budget",
                str(clean_budget),
            ],
            check=True,
        )
        print(f"Phase 1: Create {dataset} clean dataset successfully.")
    else:
        print(f"Phase 1: {dataset} clean dataset already exists. Skipping creation.")


def phase2(
    dataset,
    data_rate,
    poison_type,
    poison_rate,
    train_source,
    clean_budget,
    cover_rate=None,
):
    if poison_type in OTHER_ATTACK_POISON_TYPES:
        raise RuntimeError(
            f"{poison_type} should be handled by other_attack.py, not create_poisoned_set.py"
        )
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
        str(data_rate),
        "-train_source",
        train_source,
        "-clean_budget",
        str(clean_budget),
    ]
    if cover_rate is not None:
        cmd += ["-cover_rate", str(cover_rate)]
    print(f"=== Running Phase 2 ===")
    subprocess.run(cmd, check=True)
    if poison_type == "none":
        print(f"Phase 2: Create {data_rate} {dataset} clean dataset successfully.")
    else:
        print(
            f"Phase 2: Create {data_rate} {dataset} poisoned dataset with {poison_type} successfully."
        )


def phase3(
    dataset,
    data_rate,
    poison_type,
    poison_rate,
    num_models,
    train_source,
    clean_budget,
    cover_rate=None,
):
    if poison_type in OTHER_ATTACK_POISON_TYPES:
        raise RuntimeError(
            f"{poison_type} should be handled by other_attack.py, not train_on_poisoned_set.py"
        )
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
            "-train_source",
            train_source,
            "-clean_budget",
            str(clean_budget),
        ]
        if cover_rate is not None:
            cmd += ["-cover_rate", str(cover_rate)]
        print(f"=== Running Phase 3 ===")
        subprocess.run(cmd, check=True)
        if poison_type != "none":
            print(
                f"Phase 3: Train model {i+1}/{num_models} on {data_rate} {dataset} poisoned dataset with {poison_type} successfully."
            )
        else:
            print(
                f"Phase 3: Train model {i+1}/{num_models} on {data_rate} {dataset} clean dataset successfully."
            )


def phase_other_attack(
    dataset,
    data_rate,
    poison_type,
    poison_rate,
    num_models,
    train_source,
    clean_budget,
    cover_rate=None,
):
    for i in range(num_models):
        seed = int(time.strftime("%m%d%H%M"))
        cmd = [
            "python",
            "other_attack.py",
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
            "-train_source",
            train_source,
            "-clean_budget",
            str(clean_budget),
        ]
        if cover_rate is not None:
            cmd += ["-cover_rate", str(cover_rate)]
        print(f"=== Running Other Attack ({poison_type}) ===")
        subprocess.run(cmd, check=True)
        print(
            f"Other Attack: Train model {i+1}/{num_models} with {poison_type} on {train_source} successfully."
        )


def run_process(cfg, poison_type=None):
    dataset = cfg["dataset"]
    if poison_type is None:
        poison_type = cfg["poison_type"]
    data_rate = cfg["data_rate"]
    poison_rate = cfg["poison_rate"]
    cover_rate = None
    num_models = cfg["num_models"]
    train_source = cfg.get("train_source", "train")
    clean_budget = cfg.get("clean_budget", 2000)

    # Phase 1: Create Clean Dataset
    phase1(dataset, clean_budget)

    if poison_type in OTHER_ATTACK_POISON_TYPES:
        phase_other_attack(
            dataset,
            data_rate,
            poison_type,
            poison_rate,
            num_models,
            train_source,
            clean_budget,
            cover_rate,
        )
        return

    # Phase 2: Create Poisoned Dataset
    if poison_type in ["WaNet", "TaCT", "adaptive_blend", "adaptive_patch"]:
        cover_rate = cfg[poison_type]["cover_rate"]
    phase2(
        dataset,
        data_rate,
        poison_type,
        poison_rate,
        train_source,
        clean_budget,
        cover_rate,
    )

    # Phase 3: Train Models
    phase3(
        dataset,
        data_rate,
        poison_type,
        poison_rate,
        num_models,
        train_source,
        clean_budget,
        cover_rate,
    )


def main():
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)
    poison_types = cfg["Trainer"].get("poison_types", ["none"])
    for poison_type in poison_types:
        print(f"=== Running for poison_type: {poison_type} ===")
        run_process(cfg["Trainer"], poison_type)


if __name__ == "__main__":
    main()
