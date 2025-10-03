FROM python:3.12-slim

# ===== 构建期代理（仅构建时生效；运行期不继承）=====
ARG http_proxy=http://172.21.0.4:7899
ARG https_proxy=http://172.21.0.4:7899
ARG no_proxy=localhost,127.0.0.1,hf-mirror.com,.hf-mirror.com

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 基础工具：bash、ripgrep、dos2unix（行尾兜底）、jq（性能基准脚本用）、docker.io（仅 CLI）
RUN set -eux; \
    export http_proxy="${http_proxy}"; \
    export https_proxy="${https_proxy}"; \
    export no_proxy="${no_proxy}"; \
    apt-get update; \
    apt-get install -y --no-install-recommends bash ripgrep dos2unix jq; \
    apt-get install -y docker.io; \
    rm -rf /var/lib/apt/lists/*

# Python 依赖（项目级）
COPY requirements.txt /app/requirements.txt
RUN set -eux; \
    export http_proxy="${http_proxy}"; \
    export https_proxy="${https_proxy}"; \
    export no_proxy="${no_proxy}"; \
    pip install --no-cache-dir -r /app/requirements.txt

# 应用代码（框架层）
COPY app /app/app

# 非 root 运行用户
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /app/tmp /app/logs \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# ✅ 确保运行期 pip --user 安装即可 import（覆盖 test_pip_install 场景）
ENV PYTHONUSERBASE=/home/appuser/.local \
    PYTHONPATH=/home/appuser/.local/lib/python3.12/site-packages:$PYTHONPATH \
    PATH=/home/appuser/.local/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1

EXPOSE 8000
ENV HF_ENDPOINT=https://hf-mirror.com \
    no_proxy=localhost,127.0.0.1,hf-mirror.com,.hf-mirror.com

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ====== Task-specific payload (只保留这一段) ======
# 将任务文件复制到 /app，并在构建期一次性修正属主与执行位
USER root
WORKDIR /app

# 复制任务代码到 /app（包含 tests/、run-tests.sh、规则/日志/题目脚本等）
COPY --chown=appuser:appuser . /app/

RUN set -eux; \
    # run-tests.sh 可执行（若存在）
    if [ -f /app/run-tests.sh ]; then dos2unix /app/run-tests.sh || true; chmod 0755 /app/run-tests.sh; fi; \
    # 题目要求的脚本可执行（若存在）
    if [ -f /app/intrusion_detector.sh ]; then dos2unix /app/intrusion_detector.sh || true; chmod 0755 /app/intrusion_detector.sh; fi; \
    if [ -f /app/response.sh ]; then dos2unix /app/response.sh || true; chmod 0755 /app/response.sh; fi; \
    # 与测试工具对齐的报告目录
    mkdir -p /app/.test_results; \
    chown -R appuser:appuser /app

USER appuser
# ==================================================
