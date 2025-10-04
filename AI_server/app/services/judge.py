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
    
    ids = audience["ids"]
    embeddings = audience["embeddings"]
    metadata = audience["metadatas"]
    
    zipped = list(zip(ids, embeddings, metadata))
    zipped.sort(key = lambda x: x[2].get("index", 0))
    ids, embeddings, metadata = zip(*zipped) if zipped else ([],[],[])
    return list(ids), list(embeddings), list(metadata)

def unit_norm(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(vec) + eps
    return vec / n

# 판정 이후 청중단을 학습시킬 함수
import time
async def memorize_audience(details:list, ids:list, audience_embeddings: np.ndarray, num1_emb: np.ndarray, num2_emb: np.ndarray, *, base_learn: float = 0.03, max_learn: float = 0.10, repel: float = 0.01, clamp_norm = True,) -> None:
    """
    details: judging()에서 만든 juror별 투표/유사도 목록
    ids: chroma에 저장된 juror id 리스트 (ex. ["juror_0", ...])
    audience_embeddings: (N, D) 배열
    num1_embedding, num2_embedding: (D,)
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(current_dir, "..", "..", "data", "chroma_jurors")
    client = PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("audience")
    
    D = audience_embeddings.shape[1]
    t_now = int(time.time())
    
    
    updates_ids = []
    updates_vecs = []
    updates_metas = []
    
    N = min(len(details), len(ids), len(audience_embeddings))
    for k in range(N):
        juror_id = details[k]["juror"]
        if juror_id >= len(ids):
            continue
        
        #현재 아이디    
        cid = ids[juror_id]
        vote = details[k]["vote"]
        sim1 = float(details[k]["sim1"])
        sim2 = float(details[k]["sim2"])
        
        e = audience_embeddings[juror_id].astype(np.float32)
        # 차원이 불일치 하는 경우 에러 방지
        if e.shape[0] != D:
            continue
        
        # 타겟 + 상대 벡터
        target_vect = num1_emb if vote == "num1" else num2_emb
        other = num2_emb if vote == "num1" else num1_emb
        target = target_vect.astype(np.float32)
        other = other.astype(np.float32)
        
        # 신뢰도(margin) 기반 학습률 : margin in [0, 1]
        denom = (abs(sim1) + abs(sim2) + 1e-12)
        margin = abs(sim1 - sim2) / denom
        eta = min(max(base_learn * (0.5 + margin), base_learn), max_learn)
        
        e_new = e + eta * (target - e) - repel * (other - e)
        
        if clamp_norm:
            e_new = unit_norm(e_new)
            
        # 기존 메타데이터
        md = collection.get(ids=[cid], include=["metadatas"])
        meta = (md["metadatas"][0] if md and md.get("metadatas") else {}) or {}
        
        votes_total = int(meta.get("votes_total", 0)) + 1
        votes_num1 = int(meta.get("votes_num1", 0)) + (1 if vote == "num1" else 0)
        votes_num2 = int(meta.get("votes_num2", 0)) + (1 if vote == "num2" else 0)
        meta.update({
            "votes_total": votes_total,
            "votes_num1" : votes_num1,
            "votes_num2" : votes_num2,
            "last_conf" : round(margin, 4),
            "last_updated": t_now,
        })
        
        updates_ids.append(cid)
        updates_vecs.append(e_new.tolist())
        updates_metas.append(meta)
        
    if updates_ids:
        collection.update(
            ids = updates_ids,
            embeddings=updates_vecs,
            metadatas=updates_metas,
        )

# 학습이 제대로 되었는지 확인하는 함수
def eval_memorization_once(
    audience_embeddings: np.ndarray,
    num1_emb: np.ndarray,
    num2_emb: np.ndarray,
    voted_details: list,
):
    """
    memorize_audience 호출 전 상태에서 계산.
    반환값을 저장해두고 memorize 후 동일 함수를 다시 호출해 delta 비교.
    """
    A = audience_embeddings.astype(np.float32)
    s1 = cosine_similarity(A, num1_emb.reshape(1, -1)).ravel()
    s2 = cosine_similarity(A, num2_emb.reshape(1, -1)).ravel()

    eps = 1e-12
    margin = np.abs(s1 - s2) / (np.abs(s1) + np.abs(s2) + eps)

    # details의 vote 기준으로 선택/반대 쪽 유사도 벡터 만들기
    votes = np.array([1 if d["vote"] == "num1" else 2 for d in voted_details], dtype=np.int32)
    N = min(len(votes), len(s1))
    sel = np.where(votes[:N] == 1, s1[:N], s2[:N])
    oth = np.where(votes[:N] == 1, s2[:N], s1[:N])

    return {
        "sim1": s1,           # shape (M,)
        "sim2": s2,
        "margin": margin,
        "sel": sel,           # 선택쪽 유사도 (상위 N)
        "oth": oth,           # 반대쪽 유사도 (상위 N)
        "limit": N
    }

# 어느 정도 수치로 학습이 되고있는지 확인을 위한 함수
def summarize_delta(before: dict, after: dict):
    N = min(before["limit"], after["limit"])
    d_sel = after["sel"][:N] - before["sel"][:N]
    d_oth = after["oth"][:N] - before["oth"][:N]
    d_margin = after["margin"][:N] - before["margin"][:N]

    metrics = {
        "avg_margin_increase": float(np.mean(d_margin)),
        "pct_margin_increase": float(np.mean(d_margin > 0)),
        "pct_sel_sim_increase": float(np.mean(d_sel > 0)),
        "pct_oth_sim_decrease": float(np.mean(d_oth < 0)),
    }
    return metrics

#토론이 끝나면 summarize쪽에서 전체 요약을 받고 판정을 내릴 함수.
async def judging(summary_texts: dict, entire_data: dict = None):
    # 진영 1, 2의 전체 요약본들
    num1_text = summary_texts["num1"]
    num2_text = summary_texts["num2"]
    
    # 반박 점수는 entire에서 가져와야 함
    num1_score = 5  # 기본값
    num2_score = 5  # 기본값
    
    if entire_data:
        if "num1" in entire_data and "rebuttal_score" in entire_data["num1"]:
            num1_score = entire_data["num1"]["rebuttal_score"]
        if "num2" in entire_data and "rebuttal_score" in entire_data["num2"]:
            num2_score = entire_data["num2"]["rebuttal_score"]
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
    audience_ids, audience_embeddings, audience_metadata = await load_audience_embeddings()
    if not audience_embeddings:
        raise ValueError("청중 임베딩이 비어 있습니다. process.py를 먼저 실행하세요.")

    audience_embeddings = np.array(audience_embeddings)

    # 4. 유사도 계산
    num1_similarities = cosine_similarity([num1_embedding], audience_embeddings)[0]
    num2_similarities = cosine_similarity([num2_embedding], audience_embeddings)[0]

    # 5. soft voting 점수 계산
    eps = 1e-12
    soft_score_1 = sum([sim1 / (sim1 + sim2 + eps) for sim1, sim2 in zip(num1_similarities, num2_similarities)])
    soft_score_2 = sum([sim2 / (sim1 + sim2 + eps) for sim1, sim2 in zip(num1_similarities, num2_similarities)])
    total_soft = soft_score_1 + soft_score_2
    
    soft_ratio1 = soft_score_1 / total_soft if total_soft else 0.5
    soft_ratio2 = soft_score_2 / total_soft if total_soft else 0.5

    # 7. weighted score 계산
    alpha = 0.7  # 청중 판단 비중
    beta = 0.3   # 반박 점수 비중

    raw_weighted_score1 = alpha * soft_ratio1 + beta * norm_score1
    raw_weighted_score2 = alpha * soft_ratio2 + beta * norm_score2
    total_weight = raw_weighted_score1 + raw_weighted_score2

    # 정규화: 두 점수 비율 기반으로 투표 인원 결정
    final_ratio1 = raw_weighted_score1 / total_weight if total_weight else 0.5
    final_ratio2 = raw_weighted_score2 / total_weight if total_weight else 0.5

    num1_voting_head = round(final_ratio1 * 49)
    num2_voting_head = 49 - num1_voting_head  # 보정

    votes = {
        "num1": num1_voting_head,
        "num2": num2_voting_head,
        "none": 0
    }

    # 8. 최종 승자 판단
    if num1_voting_head > num2_voting_head:
        winner = "num1"
    elif num2_voting_head > num1_voting_head:
        winner = "num2"
    else:
        winner = "무승부"


    # 9. 각 juror 별 유사도 기록 (디버깅 및 설명용)
    voted_details = []
    for i in range(49):
        sim1 = num1_similarities[i]
        sim2 = num2_similarities[i]
        diff = sim1 - sim2
        vote = "num1" if sim1 > sim2 else "num2"
        voted_details.append({
            "juror": i,
            "vote": vote,
            "sim1": round(sim1, 5),
            "sim2": round(sim2, 5),
            "diff": round(diff, 5)
        })
        
    # 10. 로그 출력
    print(f"소프트 스코어 : num1 : {round(soft_score_1,3)} num2 : {round(soft_score_2,3)}")
    print(f"[REBUTTAL] num1: {num1_score}, num2: {num2_score}")
    print(f"[WEIGHTED raw] score1: {raw_weighted_score1:.3f}, score2: {raw_weighted_score2:.3f}")
    print(f"[WEIGHTED final ratio] score1: {final_ratio1:.3f}, score2: {final_ratio2:.3f}")
    print(f"최종 투표 수 : {num1_voting_head} vs {num2_voting_head}")

    # 11. 학습 전 벡터 저장
    before = eval_memorization_once(
        audience_embeddings=audience_embeddings,
        num1_emb=num1_embedding,
        num2_emb=num2_embedding,
        voted_details=voted_details
    )
    
    # 12. 메모리 기반 학습 업데이트
    try:
        await memorize_audience(
            details=voted_details,
            ids=audience_ids,
            audience_embeddings=audience_embeddings,
            num1_emb=num1_embedding,
            num2_emb = num2_embedding,
            base_learn=0.03,
            max_learn=0.10,
            repel=0.02,
            clamp_norm=True,
        )
    except Exception as e:
        print(f'청중 학습 실패 : {e}')

    # 13. 최신 임베딩 다시 로드 후 before와 비교해서 얼마나 개선이 되었는지 확인.
    audience_ids2, audience_embeddings2, _ = await load_audience_embeddings()
    audience_embeddings2 = np.array(audience_embeddings2)
    after = eval_memorization_once(
        audience_embeddings=audience_embeddings2,
        num1_emb=num1_embedding,
        num2_emb=num2_embedding,
        voted_details=voted_details
    )

    delta_metrics = summarize_delta(before, after)
    print("[EVAL] metrics:", delta_metrics)
    
    return {
        "winner": winner,
        "votes": votes,
        "soft_scores": {
            "num1": round(soft_score_1, 3),
            "num2": round(soft_score_2, 3)
        },
        "details": voted_details
    }