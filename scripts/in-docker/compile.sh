#!/usr/bin/env bash

# .py --> .pyc
python -m compileall .

# delete all .py
find . -type f -name "*.py" -delete