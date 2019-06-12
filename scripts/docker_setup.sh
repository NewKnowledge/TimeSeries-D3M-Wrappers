#!/bin/bash -e

git clone https://gitlab.com/jgleason/primitives
cd primitives
git checkout jg/v5.8
git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
git pull upstream master