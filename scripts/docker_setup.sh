#!/bin/bash -e

cd /
git clone https://gitlab.com/jgleason/primitives
cd primitives
git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
git pull upstream master
