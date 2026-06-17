"""智信引擎 FastAPI 应用主入口。

启动方式:
    python api/run.py
    或
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import inference, models, data, reports

# ── 创建 FastAPI 应用 ──────────────────────────────────────────────────

app = FastAPI(
    title="智信引擎 API",
    description="AI 决策可追溯平台 — 面向监管合规的 AI 可信评测 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS 配置（开发环境允许所有来源）─────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ────────────────────────────────────────────────────────────

app.include_router(inference.router)
app.include_router(models.router)
app.include_router(data.router)
app.include_router(reports.router)


# ── 全局错误处理 ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常捕获，返回统一错误格式。"""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": str(exc),
                "details": {"path": str(request.url)},
            }
        },
    )


# ── 健康检查 ────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查端点。"""
    return {"status": "ok", "service": "智信引擎 API", "version": "1.0.0"}


@app.get("/", tags=["系统"])
async def root():
    """根路径，返回 API 基本信息。"""
    return {
        "name": "智信引擎 API",
        "version": "1.0.0",
        "description": "AI 决策可追溯平台",
        "docs": "/docs",
    }
