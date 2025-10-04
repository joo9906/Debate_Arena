import os
import random
import httpx
from dotenv import load_dotenv
load_dotenv()

GMS_API_KEY = os.getenv("GMS_API_KEY")
SUMMARIZE_API_URL = os.getenv("SUMMARIZE_API_URL")

async def explain_juror_reason(summary_texts: dict, voted_details: list, winner: str):
    # 1. 해당 진영에 투표한 jurors 필터링
    jurors_voted = [d for d in voted_details if d["vote"] == ("num1" if winner == "num1" else "num2")]
    if not jurors_voted:
        return "대표 청중이 해당 입장을 지지하지 않았습니다."

    # 2. 무작위로 한 명 선택
    juror = random.choice(jurors_voted)
    juror_id = juror["juror"]
    side = juror["vote"]
    chosen_text = summary_texts[side]

    # 3. GPT 프롬프트 구성
    prompt = f"""
        당신은 AI 청중 {juror_id}번입니다. 아래는 당신이 지지한 요약문입니다:

        \"\"\"{chosen_text}\"\"\"

        본인이 어떤 사람인지 30자 이내로 설명하고 당신은 왜 이 입장을 선택했는지 3~4문장으로 설명해주세요. 
        감정, 논리, 사회적 맥락 등 개인적인 기준을 반영해 주세요. 사람처럼 자연스럽게 말해 주세요.

        """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_API_KEY}"
    }
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "당신은 투표에 참여한 AI 청중입니다. 선택 이유를 사람처럼 설명하세요."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
        response = await client.post(SUMMARIZE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    reason = result["choices"][0]["message"]["content"].strip()
    return f"대표 청중 #{juror_id}의 선택 이유:\n{reason}"
