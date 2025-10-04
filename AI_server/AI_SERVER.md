## 실행 방법
source venv/Scripts/activate
pip install - requirements
cd data
python process.py
cd ..
uvicorn main:app --reload

## 기능별 AI 모델
요청 별 사용하는 모델

논리 요약 모델 : gpt 4.1 mini
공방전 요약 모델 : gpt 4.1 mini
최종 요약 모델 : gpt 4.1
요약 임베딩 모델 : gpt 3 text embedding large
대표자 투표 이유 생성 모델 : gpt 4.1 mini (temp 0.7로 자유로운 응답)

# 폴더 내용
## app
- 메인 폴더. 서버를 돌리기 위한 파일들 정리되어 있음.

### api
- route에 api 통신을 위한 코드 모음

### core
- 현재로썬 의미있는 파일 없음

### models
- 서버를 돌리는데 필요한 input, output을 명시하기 위한 모델들 존재

### services
- AI 관련된 로직이 실제로 담긴 폴더
- ~~inference.py : 사용 전 만들어 둔 파일. 테스트용이라고 생각하면 됨~~ 삭제.
- judge : AI 청중단의 판정 로직이 들어있는 파일.
- summarize : 서버에서 들어오는 텍스트들의 요약을 진행하는 함수 파일.

## data
- ai 판정단을 돌리는데 필요한 데이터가 들어있는 폴더

### chroma_jurors
- AI 청중단 관련 파일
- chroma_jurors : gpt embedding-large를 사용해 임베딩한 청중을 저장해 둔 chroma_db
- ai_jurors : AI 청중단의 row 데이터
- ai_jurors_described.csv : ai_jurors의 청중 성향을 바탕으로 텍스트화를 진행하고 실제로 판정을 내리기 위한 description을 생성한 파일
- process.py : 데이터 전처리를 진행하고 임베딩하여 chroma_db에 저장하는 파일

## 그외
- 포트번호 8000