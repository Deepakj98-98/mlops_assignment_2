#!/bin/bash
set -eo pipefail
rm -rf package
cd python_function
python3 -m pip install --target ../package/python -r ./function/requirement.txt
