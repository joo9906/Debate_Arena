from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
import os
import torch
import httpx
import json
import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import PersistentClient
import asyncio

GMS_API_KEY = os.getenv("GMS_API_KEY")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")


# AI 청중을 통한 판정을 위해 전체 요약 텍스트를 임베딩하는 함수. 일단은 GPT-text-embedding-3-small 모델 사용.
# 크레딧을 생각보다 많이 잡아먹지 않아서 그냥 large 모델로 변경. 1536 차원 -> 3072차원
# 속도가 너무 느리면 다시 변경하거나 구글 004 모델로의 변경도 고려 중. -> 속도 괜찮아서 large 모델 유지
async def result_embedding(input_data: dict):
    text = input_data["text"]
    check_point = text.find(":")
    if check_point == -1:
        payload_text = text  # : 을 찾지 못하면 그냥 텍스트 전체를 반환해서 오류 방지
    else:
        payload_text = text[check_point:]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_API_KEY}"
    }
    
    payload = {
        "model": "text-embedding-3-large",
        "input": payload_text,
    }
    
    async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
        response = await client.post(EMBEDDING_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        output = result["data"][0]["embedding"]
    
    return output


async def load_audience_embeddings():
    # 현재 파일의 위치를 기준으로 상대 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(current_dir, "..", "..", "data", "chroma_jurors")
    
    client = PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("audience")
    audience = collection.get(include=["embeddings", "metadatas"])
    
    embeddings = audience["embeddings"]
    metadata = audience["metadatas"]
    
    return embeddings, metadata
    
#토론이 끝나면 summarize쪽에서 전체 요약을 받고 판정을 내릴 함수.
async def judging(summary_texts: dict):
    # 진영 1, 2의 전체 요약본들
    num1_text = summary_texts["num1"]["text"] if isinstance(summary_texts["num1"], dict) else summary_texts["num1"]
    num2_text = summary_texts["num2"]["text"] if isinstance(summary_texts["num2"], dict) else summary_texts["num2"]
    
    # 진영 1, 2의 반박 점수 평균
    num1_score = summary_texts["num1"]["rebuttal_score"]
    num2_score = summary_texts["num2"]["rebuttal_score"]
    # 10으로 나눠 정규화
    norm_score1 = num1_score / 10
    norm_score2 = num2_score / 10
    # 스코어와 판정을 기반으로 임의로 판정단의 수를 만들거임.
    
    
    print("[DEBUG] 1번 주장 요약문:", num1_text, "점수 : ", norm_score1)
    print("[DEBUG] 2번 주장 요약문:", num2_text, "점수 : ", norm_score2)

    # 2. 요약 임베딩
    num1_embedding, num2_embedding = await asyncio.gather(
        result_embedding({"text": num1_text}),
        result_embedding({"text": num2_text})
    )
    num1_embedding = np.array(num1_embedding)
    num2_embedding = np.array(num2_embedding)

    # 3. 청중 임베딩 불러오기, 오류 방지 들어감
    audience_embeddings, audience_metadata = await load_audience_embeddings()
    if audience_embeddings is None or len(audience_embeddings) == 0:
        raise ValueError("청중 임베딩이 비어 있습니다. process.py를 먼저 실행하세요.")

    sorted_data = sorted(zip(audience_embeddings, audience_metadata), key=lambda x: x[1].get("id", 0))
    audience_embeddings = np.array([x[0] for x in sorted_data])

    # 4. 유사도 계산
    num1_similarities = cosine_similarity([num1_embedding], audience_embeddings)[0]
    num2_similarities = cosine_similarity([num2_embedding], audience_embeddings)[0]

    # 5-1. hard voting (MARGIN 기반)
    MARGIN = 0.01
    votes = {"num1": 0, "num2": 0, "none": 0} # none = 기권
    voted_details = []

    for i in range(len(audience_embeddings)):
        sim1 = num1_similarities[i]
        sim2 = num2_similarities[i]
        diff = sim1 - sim2

        if diff > MARGIN:
            votes["num1"] += 1
            vote = "num1"
        elif diff < -MARGIN:
            votes["num2"] += 1
            vote = "num2"
        else:
            votes["none"] += 1
            vote = "none"

        voted_details.append({
            "juror": i,
            "vote": vote,
            "sim1": round(sim1, 5),
            "sim2": round(sim2, 5),
            "diff": round(diff, 5)
        })

    # 5-2. soft voting 점수 계산
    soft_score_1 = 0
    soft_score_2 = 0
    for sim1, sim2 in zip(num1_similarities, num2_similarities):
        soft_score_1 += round(sim1 / (sim1 + sim2))
        soft_score_2 += round(sim2 / (sim1 + sim2))

    # 6. 최종 승자 판단 (하드 기준 우선)
    if votes["num1"] > votes["num2"]:
        winner = "num1"
    elif votes["num2"] > votes["num1"]:
        winner = "num2"
    else:
        # 하드 동점이면 soft로 결정
        if soft_score_1 > soft_score_2:
            winner = summary_texts["num1"] + " (soft)"
        elif soft_score_2 > soft_score_1:
            winner = summary_texts["num2"] + " (soft)"
        else:
            winner = "무승부"

    # 로그 출력
    print(f"[HARD] num1: {votes['num1']} / num2: {votes['num2']} / 기권: {votes['none']}")
    print(f"[SOFT] score1: {soft_score_1:.2f} / score2: {soft_score_2:.2f}")

    # 반환값
    return {
        "winner": winner,
        "votes": votes,
        "soft_scores": {
            "num1": round(soft_score_1, 3),
            "num2": round(soft_score_2, 3)
        },
        "details": voted_details
    }
