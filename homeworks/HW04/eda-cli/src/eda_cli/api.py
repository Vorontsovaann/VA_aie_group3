# src/eda_cli/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import time
from .core import compute_quality_flags, missing_table, DatasetSummary

app = FastAPI(title="EDA Quality API", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/quality-from-csv")
async def quality_from_csv(file: UploadFile = File(...)):
    """Оригинальный эндпоинт из семинара — можно оставить как есть."""
    start = time.time()
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        summary = DatasetSummary(df)
        duration = time.time() - start
        return {
            "summary": summary.to_dict(),
            "duration_seconds": round(duration, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(file: UploadFile = File(...)):
    """
    НОВЫЙ ЭНДПОИНТ (HW04):
    Возвращает все флаги качества, включая те, что добавлены в HW03.
    """
    start = time.time()
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Вычисляем флаги качества (включая ваши новые!)
        flags = compute_quality_flags(df)

        # Убираем не-JSON-сериализуемые типы (если есть)
        # В нашем случае все значения — bool/float/int — OK

        duration = time.time() - start

        return JSONResponse(content={
            "flags": flags,
            "duration_seconds": round(duration, 3)
        })

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")