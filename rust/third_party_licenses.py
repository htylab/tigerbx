# Writes THIRD_PARTY_LICENSES.txt for the executable: every Rust crate compiled into it for one target (normal and
# build dependencies, from `cargo metadata --locked`) with its license expression and the license / notice files it
# ships, then ONNX Runtime's LICENSE and ThirdPartyNotices.txt.
#   python third_party_licenses.py <target triple> <ONNX Runtime LICENSE> <ThirdPartyNotices.txt> <out file>
import json
import os
import subprocess
import sys

NOTICE_PREFIXES = ('LICENSE', 'LICENCE', 'COPYING', 'NOTICE', 'UNLICENSE', 'COPYRIGHT')
RULE = '=' * 100


def crates(target):
    meta = json.loads(subprocess.run(['cargo', 'metadata', '--format-version', '1', '--locked', '--filter-platform',
                                      target], check=True,
                                     stdout=subprocess.PIPE).stdout.decode('utf-8'))   # Python 3.6: Linux container
    nodes = {n['id']: n for n in meta['resolve']['nodes']}
    packages = {p['id']: p for p in meta['packages']}
    seen, stack = set(), [meta['resolve']['root']]
    while stack:
        for d in nodes[stack.pop()]['deps']:
            if d['pkg'] not in seen and any(k['kind'] in (None, 'build') for k in d['dep_kinds']):
                seen.add(d['pkg'])
                stack.append(d['pkg'])
    return sorted((packages[i] for i in seen), key=lambda p: (p['name'], p['version']))


def notice_files(p):
    root = os.path.dirname(p['manifest_path'])
    files = sorted(f for f in os.listdir(root) if f.upper().startswith(NOTICE_PREFIXES)
                   and os.path.isfile(os.path.join(root, f)))
    if p.get('license_file') and os.path.basename(p['license_file']) not in files:
        files.append(p['license_file'])
    return [os.path.join(root, f) for f in files]


def main(target, ort_license, ort_notices, out):
    pkgs = crates(target)
    missing = [p['name'] for p in pkgs if not notice_files(p)]
    if missing:
        sys.exit(f'no license file in: {", ".join(missing)}')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('Third-party software in the tigerbx executable\n\n'
                 'TigerBx itself and its models are licensed under CC BY-NC-SA 4.0 (LICENSE). The executable also\n'
                 'contains the open-source components below, each under its own license.\n\n')
        fh.write(f'Rust crates ({len(pkgs)}, target {target}):\n')
        for p in pkgs:
            fh.write(f'  {p["name"]} {p["version"]}  {p["license"] or "see license file"}\n')
        fh.write('  ONNX Runtime  MIT (Microsoft)\n')
        for p in pkgs:
            fh.write(f'\n{RULE}\n{p["name"]} {p["version"]}  ({p["license"] or "see license file"})'
                     f'{"  " + p["repository"] if p.get("repository") else ""}\n{RULE}\n')
            for f in notice_files(p):
                fh.write(f'\n--- {os.path.basename(f)} ---\n')
                fh.write(open(f, encoding='utf-8', errors='replace').read().rstrip() + '\n')
        for name, f in (('ONNX Runtime LICENSE', ort_license), ('ONNX Runtime ThirdPartyNotices.txt', ort_notices)):
            fh.write(f'\n{RULE}\n{name}\n{RULE}\n\n')
            fh.write(open(f, encoding='utf-8', errors='replace').read().rstrip() + '\n')
    print(f'{out}: {len(pkgs)} crates + ONNX Runtime')


if __name__ == '__main__':
    main(*sys.argv[1:5])
