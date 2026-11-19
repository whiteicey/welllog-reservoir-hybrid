#!/usr/bin/env bash
# Push local code (top-level scripts + package/) to the server v3 dir (no data, no outputs).
set -e
cd "$(dirname "$0")"
shopt -s nullglob
files=(*.py *.sh *.json *.md *.txt package)
tar czf /tmp/v3push.tgz --exclude='__pycache__' "${files[@]}"
scp -q /tmp/v3push.tgz <SERVER_ALIAS_TLS>:<PROJECT_ROOT>
ssh -o BatchMode=yes <SERVER_ALIAS_TLS> 'cd <PROJECT_ROOT> && tar xzf v3push.tgz && rm v3push.tgz && sed -i "s/\r$//" *.sh *.py package/*.py package/src/*.py package/*.yaml package/*.md package/*.txt'
rm -f /tmp/v3push.tgz
