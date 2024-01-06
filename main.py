import datetime
import httpx
import paddleocr

from io import BytesIO
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.status import *


def PaddleOCR(content, lang="ch") -> str:
    def update_ch_model():
        paddleocr.paddleocr.MODEL_URLS["OCR"]["PP-OCRv4"]["det"]["ch"][
            "url"
        ] = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar"

        paddleocr.paddleocr.MODEL_URLS["OCR"]["PP-OCRv4"]["rec"]["ch"][
            "url"
        ] = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar"

    try:
        image = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(
            detail="解析图像文件失败",
            status_code=HTTP_400_BAD_REQUEST,
        )

    ocr_object = paddleocr.PaddleOCR(
        lang=lang,
        show_log=False,
        use_angle_cls=True,
    )

    lines = []

    try:
        r = ocr_object.ocr(content)

        for data_lines in r:
            cur_height = 0
            cur_data = []

            for data in data_lines:
                height_up = min([x[1] for x in data[0]])
                height_down = max([x[1] for x in data[0]])

                if height_up > cur_height:
                    if cur_data:
                        lines.append(" ".join(cur_data))
                        cur_data = []

                cur_data.append(data[1][0].strip())
                cur_height = height_down

            if cur_data:
                lines.append(" ".join(cur_data))
    except Exception:
        pass

    return "\n".join(lines)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/ocr",
    response_model=str,
    summary="OCR(输入图像URL)",
)
async def ocr(request: Request) -> str:
    url = request.query_params.get("url")

    if not url:
        raise HTTPException(
            detail="参数错误(需要图像URL)",
            status_code=HTTP_400_BAD_REQUEST,
        )

    content = None

    try:
        r = httpx.get(url)

        if r.status_code == 200:
            content = r.content
    except Exception:
        pass

    if content is None:
        raise HTTPException(
            detail="获取图像文件失败",
            status_code=HTTP_400_BAD_REQUEST,
        )

    return PaddleOCR(content)


@app.post(
    "/ocr_content",
    response_model=str,
    summary="OCR(输入图像内容)",
)
async def ocr_content(request: Request) -> str:
    content = await request.body()

    return PaddleOCR(content)
