#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script

if [ ${1} == 'coverage' ]
then
  pytest --cov=../autodiff_NARS --cov-fail-under=90 --cov-report html:cov_html
elif [ ${1} == 'test' ]
then
  pytest
fi
