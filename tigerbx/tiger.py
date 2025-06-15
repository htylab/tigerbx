import argparse
import sys
from tigerbx import bx
from tigerbx import gdmi
from tigerbx import hlc171
from tigerbx import reg_vbm
from tigerbx import nerve_nerme

def main():
    parser = argparse.ArgumentParser(prog="tiger", description="Tiger CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # BX subcommand
    bx_parser = subparsers.add_parser("bx", help="Run bx module")
    bx.setup_parser(bx_parser)

    # GDM subcommand
    gdm_parser = subparsers.add_parser("gdm", help="Run gdm module")
    gdmi.setup_parser(gdm_parser)

    # HLC subcommand
    hlc_parser = subparsers.add_parser("hlc", help="Run hlc module")
    hlc171.setup_parser(hlc_parser)
    
    # REG subcommand
    reg_parser = subparsers.add_parser("reg", help="Run reg module")
    reg_vbm.setup_parser(reg_parser)

    # NERVE subcommand
    reg_parser = subparsers.add_parser("nerve", help="Run NERVE module")
    nerve_nerme.setup_parser(reg_parser)




    args = parser.parse_args()

    if args.command == "bx":
        bx.run_args(args)
    elif args.command == "gdm":
        gdmi.run_args(args)
    elif args.command == "hlc":
        hlc171.run_args(args)
    elif args.command == "reg":
        reg_vbm.run_args(args)
    elif args.command == "nerve":
        nerve_nerme.run_args(args)

if __name__ == "__main__":
    main()