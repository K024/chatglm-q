#!/bin/bash
set -euxo pipefail

wget https://cloud.tsinghua.edu.cn/seafhttp/files/1185de71-73f3-4355-bba4-56d0e6858917/CEval.zip
unzip -q CEval.zip -x '__MACOSX/*'
rm -f CEval.zip
