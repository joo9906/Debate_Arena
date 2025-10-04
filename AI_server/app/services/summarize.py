from transformers import pipeline
from app.models.schemas import STTRequest, RequestInput, ResponseOutput, LastInput, LastOutput, AttackdefenseInput
import os
import httpx
from dotenv import load_dotenv
load_dotenv()

GMS_API_KEY = os.getenv("GMS_API_KEY")
SUMMARIZE_API_URL = os.getenv("SUMMARIZE_API_URL")
    
# 처음으로 들어온 텍스트 요약. 공방 이전에 개인의 논리 주장 시 사용하는 함수.
async def summarize_first_half(input_data: STTRequest) -> str:
    user_id = input_data.user_id
    topic = input_data.topic
    text = input_data.text
    position = input_data.position
    max_text_len = round(len(text)/2)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {GMS_API_KEY}"
    }
    
    prompt = f"""
    {topic}에서 {position}의 역할인 사람이 말한 내용은 다음과 같습니다: {text}.
    
    이 발언의 핵심 주장을 한국어로 반드시 최소 150자 이상, 반드시 최대 {max_text_len}자 이내로 요약해 주세요.
    150자를 채우기 위해 없는 내용을 추가하지 말고 최대한 주어진 텍스트 내에서 가공하여 내용의 핵심을 요약해 주세요.
    공격적인 발언이 있는 경우 비속어가 아니라면 요약본에 무조건 추가해주세요.
    """
    
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "한국어로 대답해주세요. 앞뒤로 토론과 관련 없는 내용은 넣지 말고 간략하게 요약해 주세요."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }
    
    # 5초간 대답이 없으면 에러로 간주. 
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        response = await client.post(SUMMARIZE_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 에러 발생 시 예외 던짐
        result = response.json()     # JSON 파싱

    # 응답 결과에서 요약 추출
    result = {"user_id" : user_id, "text": result["choices"][0]["message"]["content"].strip()}
    
    return result
    

# 공방전에서 들어온 텍스트 요약
async def summarize_second_half(input_data: AttackdefenseInput):
    topic = input_data.topic
    data = input_data.key
    
    attack_id = data["attack"]["user_id"]
    attack_position = data["attack"]["position"]
    attack_text = data["attack"]["text"]
    attack_len = len(attack_text)
    
    defense_id = data["defense"]["user_id"]
    defense_position = data["defense"]["position"]
    defense_text = data["defense"]["text"]
    defense_len = len(defense_text)

    headers = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {GMS_API_KEY}"
    }
    
    prompt = f"""
    {topic}에 대한 공방전입니다.

    - {attack_position}을 주장하는 사람이 아래와 같은 내용으로 공격했습니다:
    "{attack_text}"

    - 이에 대해 {defense_position}을 주장하는 사람이 아래와 같은 내용으로 방어했습니다:
    "{defense_text}"

    1. 위 내용을 각각 요약해 주세요. 요약 형식은 다음과 같이 해주세요:

    공격 내용 : (공격 요약, 최대 분량은 {attack_len/2}글자 내로)
    방어 내용 : (방어 요약, 최대 분량은 {defense_len/2}글자 내로)

    2. 그리고 방어자의 반박이 공격자의 논리를 얼마나 효과적으로 반박했는지 0~10점으로 판단해주세요.
    오직 숫자만 출력하지 말고, 아래 형식으로 작성해주세요:

    반박 점수 : X점
    """
    
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "한국어로 대답해주세요. 간략하고 핵심적인 요약을 해주세요. "},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.3
    }
    
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        print(f'{topic}에 대한 {attack_text} 공방전 요약 시작')
        response = await client.post(SUMMARIZE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
    
        combined_summarize = result["choices"][0]["message"]["content"].strip()
    
    # 요약본에서 반박 점수만 빼오는 함수
    import re
    def extract_rebuttal_score(llm_text: str) -> int:
        match = re.search(r"반박 점수\s*:\s*(\d+)", llm_text)
        if match:
            return int(match.group(1))
        return 5
    
    content = {
        "attack_id": attack_id,
        "defense_id": defense_id,
        "text": combined_summarize,
        "rebuttal_score": extract_rebuttal_score(combined_summarize)
    }

    return {"result": content}


# 토론이 끝나면 모든 요약본을 받아서 전체 요약을 받아올 함수.
async def summarize_result_text(input_data: LastInput):
    topic = input_data.topic
    entire = input_data.entire
    
    summaries = {}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {GMS_API_KEY}"
    }
    
    for key, val in entire.items():
        position = val["position"]
        text = val["text"]
    
        prompt = f"""
        당신은 전문적인 토론 분석가입니다. 아래는 {topic}에 대해 {position}을 주장하는 한 명 이상의 인물들이 참여한 토론의 주장 요약입니다.

        다음은 요약할 토론 내용입니다: {text}
         
        이 토론을 해당 인물이 주장하고자 하는 바를 반드시 최소 400자 이상으로 요약해주세요.
        불필요한 잡담, 반복, 감정 표현은 제거하고 핵심 주장과 논리적 구조 중심으로 작성해 주세요.
        topic과 position은 가능한 적게 사용하면서 답변을 생성 해주세요.
        해당 요약본은 임베딩하여 AI 청중단과 코사인 유사도를 비교할 것입니다. 
        최대 토큰 내에서 적절히 분량을 조절해서 결과를 만들어주세요.

        """
        
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "developer", "content": "한국어로 대답해주세요."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.1
        }
        
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            print("최종 요약 시작")
            response = await client.post(SUMMARIZE_API_URL, headers=headers, json=payload)
            response.raise_for_status()  # 에러 발생 시 예외 던짐
            result = response.json()     # JSON 파싱
        
        summaries[key] = result["choices"][0]["message"]["content"].strip()

    return summaries

