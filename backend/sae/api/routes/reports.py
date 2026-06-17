"""报告生成路由 — POST /v1/reports/generate"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from api.schemas.request import ReportGenerateRequest
from api.schemas.response import ErrorResponse, ReportGenerateResponse
from api.services.report_service import generate_report

router = APIRouter(prefix="/v1/reports", tags=["报告生成"])


@router.post(
    "/generate",
    response_model=ReportGenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数错误"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
    },
    summary="生成报告",
    description="生成合规溯源报告，支持 PDF/HTML 格式。",
)
async def generate_report_endpoint(req: ReportGenerateRequest):
    """生成统一归因报告。

    使用 task_d_report 的 report_schema 和 report_template 渲染。
    对于 PDF 格式，当前返回 HTML（PDF 转换可后续添加）。
    """
    if not req.trace_ids:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "invalid_params", "message": "trace_ids 不能为空"}},
        )

    try:
        result = generate_report(
            trace_ids=req.trace_ids,
            fmt=req.format,
            template_name=req.template,
            include_visualizations=req.include_visualizations,
        )
        return ReportGenerateResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "report_error", "message": str(e)}},
        )
