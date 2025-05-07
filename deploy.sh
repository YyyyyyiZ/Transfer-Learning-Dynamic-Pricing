#!/usr/bin/env sh
set -e

cd docs  # 直接进入docs目录
git init
git add -A
git commit -m 'deploy'
git push -f git@github.com:YyyyyyiZ/Transfer-Learning-Dynamic-Pricing.git master:gh-pages