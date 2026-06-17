"""智信引擎 API 启动脚本。

用法:
    python api/run.py
"""

import sys
import os

# 确保 SAE 根目录在 Python 路径中
SAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SAE_ROOT not in sys.path:
    sys.path.insert(0, SAE_ROOT)

import uvicorn


def main():
    """启动 FastAPI 服务。"""
    print("=" * 50)
    print("  智信引擎 API 服务")
    print("  http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
