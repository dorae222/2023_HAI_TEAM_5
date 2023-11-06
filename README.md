# 2023_HAI_TEAM_5
2023년 HAI 하반기 5팀 프로젝트 전용 REPO입니다.

# 기본 환경 세팅
1. (anaconda prompt) conda create -n "hai" python=3.10
2. (anaconda prompt) conda activate hai
3. (anaconda prompt) pip install -r requirements.txt

# Git 명령어

## 버전 생성
1. git 현재 상태 확인 : git status
2. git 버전 추적: git add .
3. git 버전 생성: git commit -m"텍스트"
4. git 버전 확인: git log

## 브랜치 관리
1. 브랜치 목록 확인: git branch
2. 브랜치 이동: git checkout 브랜치이름
3. 브랜치 내 push: git push origin 브랜치 이름
   
## 버전 되돌리기
1. 이전 버전으로 돌아가기: git reset --hard 브랜치2~
2. 돌아왔던 버전으로 되돌아가기: git reset --hard 브랜치3
3. 특정 버전으로 되돌아가기: git log → git reset --hard 툭정_버전_ID
