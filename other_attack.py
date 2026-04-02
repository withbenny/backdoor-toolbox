import argparse
import os
import sys

import config
from utils import default_args, supervisor


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dataset",
        type=str,
        required=False,
        default=default_args.parser_default["dataset"],
        choices=default_args.parser_choices["dataset"],
    )
    parser.add_argument(
        "-poison_type",
        type=str,
        required=False,
        choices=default_args.parser_choices["poison_type"],
        default=default_args.parser_default["poison_type"],
    )
    parser.add_argument(
        "-poison_rate",
        type=float,
        required=False,
        choices=default_args.parser_choices["poison_rate"],
        default=default_args.parser_default["poison_rate"],
    )
    parser.add_argument(
        "-cover_rate",
        type=float,
        required=False,
        choices=default_args.parser_choices["cover_rate"],
        default=default_args.parser_default["cover_rate"],
    )
    parser.add_argument(
        "-alpha",
        type=float,
        required=False,
        default=default_args.parser_default["alpha"],
    )
    parser.add_argument("-test_alpha", type=float, required=False, default=None)
    parser.add_argument("-trigger", type=str, required=False, default=None)
    parser.add_argument("-no_aug", default=False, action="store_true")
    parser.add_argument("-model", type=str, required=False, default=None)
    parser.add_argument("-model_path", required=False, default=None)
    parser.add_argument("-no_normalize", default=False, action="store_true")
    parser.add_argument("-devices", type=str, default="0")
    parser.add_argument("-log", default=False, action="store_true")
    parser.add_argument("-seed", type=int, required=False, default=default_args.seed)
    parser.add_argument("-data_rate", type=float, required=False, default=1.0)
    parser.add_argument("-clean_budget", type=int, required=False, default=2000)
    parser.add_argument(
        "-train_source",
        type=str,
        required=False,
        default="train",
        choices=["train", "test"],
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.trigger is None:
        args.trigger = config.trigger_default[args.dataset][args.poison_type]

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
    if args.log:
        out_path = "logs"
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = os.path.join(out_path, "%s_seed=%s" % (args.dataset, args.seed))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = os.path.join(out_path, "other_attack")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        out_path = os.path.join(
            out_path,
            "%s_%s.out"
            % (
                args.poison_type,
                supervisor.get_dir_core(
                    args,
                    include_model_name=True,
                    include_poison_seed=config.record_poison_seed,
                ),
            ),
        )
        fout = open(out_path, "w")
        ferr = open(os.devnull, "a")
        sys.stdout = fout
        sys.stderr = ferr

    if args.poison_type == "trojannn":
        from other_attacks_tool_box import trojannn

        attacker = trojannn.attacker(args)
        attacker.attack()
    elif args.poison_type == "BadEncoder":
        from other_attacks_tool_box import BadEncoder

        attacker = BadEncoder.attacker(args)
        attacker.attack()
    elif args.poison_type == "SRA":
        from other_attacks_tool_box import SRA

        attacker = SRA.attacker(args)
        attacker.attack()
    elif args.poison_type == "bpp":
        from other_attacks_tool_box import bpp

        attacker = bpp.attacker(args)
        attacker.attack()
    elif args.poison_type == "WB":
        from other_attacks_tool_box import WB

        attacker = WB.attacker(args)
        attacker.attack()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
