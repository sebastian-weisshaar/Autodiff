#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script

pytest --cov=../src/autodiff --cov-fail-under=90 --cov-report html:cov_html