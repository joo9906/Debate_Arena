from chromadb import PersistentClient
import pandas as pd
import os
import requests
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv("ai_jurors.csv")
GMS_API_KEY = os.getenv("GMS_API_KEY")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
chroma_path = "chroma_jurors" # Chroma 저장 공간

# 헷갈려서 class로 묶어버림. 이건 청중의 성향을 자연어로 바꾸는 함수 모음.
class process_csv:
    # 각 성향 항목을 자연어로 바꾸는 함수들
    def describe_openness(self, val):
        return "창의적이고 새로운 것을 좋아해요" if val > 0.5 else "새로운 것보다는 익숙한 걸 선호해요"

    def describe_conscientiousness(self, val):
        return "계획적이고 신중한 성격이에요" if val > 0.5 else "즉흥적이고 자유로운 성향이에요"

    def describe_extraversion(self, val):
        return "사교적이고 활발한 성격이에요" if val > 0.5 else "조용하고 혼자 있는 걸 좋아해요"

    def describe_agreeableness(self, val):
        return "공감하고 협력하는 걸 중요하게 여겨요" if val > 0.5 else "비판적으로 생각하고 자기 주장이 강해요"

    def describe_neuroticism(self, val):
        return "감정 기복이 심하고 예민한 편이에요" if val > 0.5 else "감정이 안정적이고 침착한 편이에요"

    def describe_age(self, age):
        if age < 20:
            return "10대예요"
        elif age < 30:
            return "20대예요"
        elif age < 40:
            return "30대예요"
        elif age < 50:
            return "40대예요"
        else:
            return "50대 이상이에요"

    def describe_gender(self, gender):
        return "남성이에요" if gender == 0 else "여성이에요"

    # 하나의 자연어 설명으로 합치는 함수
    def create_description(self, row):
        traits = [
            self.describe_age(row["Age"]),
            self.describe_gender(row["Gender"]),
            self.describe_openness(row["Openness"]),
            self.describe_conscientiousness(row["Conscientiousness"]),
            self.describe_extraversion(row["Extraversion"]),
            self.describe_agreeableness(row["Agreeableness"]),
            self.describe_neuroticism(row["Neuroticism"]),
        ]
        return " ".join(traits)

# description 컬럼을 추가하고 개인의 성향을 자연어로 설명
process = process_csv()
df["description"] = df.apply(process.create_description, axis=1)

# api로 임베딩을 실행할 함수. 청중의 성향 description을 이 함수를 사용해서 임베딩 할 거임.
def description_embedding(text:str) -> list:
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {GMS_API_KEY}"
    }
    
    payload = {
        "model": "text-embedding-3-large",
        "input": text,
    }
    
    response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    output = result["data"][0]["embedding"]
        
    return output

# csv 파일에서 description만 추출 후 임베딩 실행.
description = df["description"].tolist()
description_list = [description_embedding(desc) for desc in description]

# 가짜 벡터 삽입을 위한 준비. 나는 GMS로 만들어진걸 사용하기 때문에 빈껍데기라도 넣어서 chroma_db를 통과해야 함.
fake_embeddings = FakeEmbeddings(size=len(description_list[0]))

# 실제 임베딩에 사용할 documents
documents = [
    Document(
        page_content=row["description"],
        metadata={
            "age": int(row["Age"]),
            "gender": "남성" if row["Gender"] == "남성" else "여성",
            "index": i
        }
    ) for i, row in df.iterrows()
]

# 크로마db에 백터 수동으로 업데이트
chroma_db = Chroma.from_documents(
    documents=documents, 
    embedding = fake_embeddings, 
    persist_directory=chroma_path)

# 실제 임베딩으로 업데이트
client = PersistentClient(path=chroma_path)
collection = client.get_or_create_collection("audience")

# 기존 데이터 삭제 - 모든 데이터를 삭제하기 위해 where 조건 사용
try:
    # 모든 데이터를 삭제하기 위해 임의의 조건 사용
    collection.delete(where={"index": {"$gte": 0}})
except:
    # 조건이 실패하면 컬렉션을 다시 생성
    client.delete_collection("audience")
    collection = client.create_collection("audience")

# 실제 임베딩과 메타데이터 추가
for i, (desc, embedding) in enumerate(zip(description, description_list)):
    collection.add(
        ids=[f"juror_{i}"],
        embeddings=[embedding],
        documents=[desc],
        metadatas=[{
            "age": int(df.iloc[i]["Age"]),
            "gender": "남성" if df.iloc[i]["Gender"] == 0 else "여성",
            "index": i
        }]
    )

print(f"총 {len(description_list)}개의 청중 임베딩이 저장되었습니다.")

results = chroma_db.similarity_search("논리적인 설명을 선호해요", k=3)
for r in results:
    print("내용:", r.page_content)
    print("메타데이터:", r.metadata)
    print("-" * 30)

print("차원:", len(description_list[0])) 