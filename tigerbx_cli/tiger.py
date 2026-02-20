import sys
import argparse
from tigerbx_cli import bx_cli, gdm_cli, hlc_cli, reg_cli, nerve_cli

_BX_SHORT_FLAGS = set('bmacCdSWtqzpg')


def _expand_bx_flags(argv):
    """Expand combined short flags for the bx subcommand.

    e.g. ['-bmad'] -> ['-b', '-m', '-a', '-d']
    Only expands tokens where every character is a known bx single-char flag.
    Long options (--...) and unknown combinations are left untouched.
    """
    result = []
    for arg in argv:
        if (arg.startswith('-') and not arg.startswith('--')
                and len(arg) > 2
                and all(c in _BX_SHORT_FLAGS for c in arg[1:])):
            result.extend(f'-{c}' for c in arg[1:])
        else:
            result.append(arg)
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'bx':
        sys.argv = [sys.argv[0], 'bx'] + _expand_bx_flags(sys.argv[2:])

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
