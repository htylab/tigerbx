import argparse
from tigerbx_cli import bx_cli, gdm_cli, hlc_cli, reg_cli, nerve_cli


def main():
    parser = argparse.ArgumentParser(prog="tiger", description="Tiger CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=False)

    bx_cli.setup_parser(subparsers.add_parser("bx", help="Run bx module"))
    gdm_cli.setup_parser(subparsers.add_parser("gdm", help="Run gdm module"))
    hlc_cli.setup_parser(subparsers.add_parser("hlc", help="Run hlc module"))
    reg_cli.setup_parser(subparsers.add_parser("reg", help="Run reg module"))
    nerve_cli.setup_parser(subparsers.add_parser("nerve", help="Run NERVE module"))

    args = parser.parse_args()

    dispatch = {
        "bx": bx_cli,
        "gdm": gdm_cli,
        "hlc": hlc_cli,
        "reg": reg_cli,
        "nerve": nerve_cli,
    }
    if args.command in dispatch:
        dispatch[args.command].run_args(args)


if __name__ == "__main__":
    main()
