FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（部分 Python 包需要编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 复制项目代码
COPY . .

# 默认命令
CMD ["researchbot", "agent"]
