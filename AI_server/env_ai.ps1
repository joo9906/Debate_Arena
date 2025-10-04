Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\ai\Scripts\activate
# pip 업데이트
python -m pip install --upgrade pip

pip install -r requirements.txt
