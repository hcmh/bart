#!/bin/bash

echo 'VERSION('`./git-version.sh`')' > version.new
./update-if-changed.sh version.new src/misc/version.inc
rm -f version.new

