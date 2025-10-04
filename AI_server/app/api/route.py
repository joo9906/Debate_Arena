from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from app.models.schemas import RequestInput, ResponseOutput, STTRequest, EmbeddingOutput, AttackdefenseInput, AttackDefenseOutput,LastOutput, LastInput
from app.services.summarize import summarize_first_half, summarize_result_text, summarize_second_half
from app.services.judge import judging
from app.services.rag_explain import explain_juror_reason

router = APIRouter()

@router.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"

# 1차로 의견 주장하며 받아온 텍스트를 요약본으로 바꾸는 곳
@router.post("/summaries/opinion", response_model=ResponseOutput)
async def summarize_opinion(input_data: STTRequest):
    result = await summarize_first_half(input_data)
    return {"result" : result}

# 공방전을 하며 나온 공격/방어를 받아서 요약본으로 바꿔주는 곳
@router.post("/summaries/seigedefense", response_model=AttackDefenseOutput)
async def summarize_seige_defense(input_data: AttackdefenseInput):
    result = await summarize_second_half(input_data)
    return result

# 최종으로 모든 요약을 받아서 결과를 도출하는 곳
@router.post("/summaries/result", response_model=LastOutput)
async def last(input_data: LastInput):
    draw = input_data.draw
    # 1. 요약 텍스트 생성
    summary_texts = await summarize_result_text(input_data)
    
    # 무승부인 경우 
    if draw == True:
        # 2. judge.py의 judging 함수 호출하여 최종 판정 (entire_data 전달)
        result = await judging(summary_texts, input_data.entire)
        # 3. 임의의 판정단 한 명을 선택하여 승자에게 투표한 이유를 설명해 줌.
        explain_result = await explain_juror_reason(summary_texts, result["details"], result["winner"])
        
        # 결과 + 청중단 판정 이유 + 전체 요약
        return {"result": result, "juror_explain": explain_result, "full_summarize": summary_texts, }
    else:
        # 무승부가 아니라면 전체 요약만 던져 줌
        return {"full_summarize": summary_texts}