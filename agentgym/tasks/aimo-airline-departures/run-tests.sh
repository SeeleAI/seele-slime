#!/usr/bin/env bash
set -euo pipefail

# 统一的工作目录与测试目录
WORKDIR="/app"
TESTS_DIR="/app/tests"

# 与 TestTool 默认一致的报告目录
RESULTS_DIR="${WORKDIR}/.test_results"
JUNIT_XML="${RESULTS_DIR}/junit.xml"

# 可选：透传额外 pytest 参数（比如 -rA、-k 关键字等）
PYTEST_ARGS="${PYTEST_ARGS:-}"

mkdir -p "${RESULTS_DIR}"

# 执行测试
python -m pytest -q --disable-warnings --maxfail=1 \
  --junitxml="${JUNIT_XML}" \
  ${PYTEST_ARGS} \
  "${TESTS_DIR}"
