import argparse
import sys
from tigerbx import bx
from tigerbx import gdm

def main():
    parser = argparse.ArgumentParser(prog="tiger", description="Tiger CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # BX subcommand
    bx_parser = subparsers.add_parser("bx", help="Run bx module")
    bx.setup_parser(bx_parser)

    # GDM subcommand
    gdm_parser = subparsers.add_parser("gdm", help="Run gdm module")
    gdm.setup_parser(gdm_parser)


    args = parser.parse_args()

    if args.command == "bx":
        bx.run_args(args)
    elif args.command == "gdm":
        gdm.run_args(args)

if __name__ == "__main__":
    main()