@echo off
chcp 65001 > nul

echo ============================================
echo AI 음식 레시피 챗봇 설치를 시작합니다.
echo ============================================
echo.

echo [1/3] Python 설치 여부 확인 중...
python --version

IF ERRORLEVEL 1 (
    echo.
    echo Python이 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo Python 3.11 이상 설치 후 다시 실행해주세요.
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b
)

echo.
echo [2/3] pip 업그레이드 중...
python -m pip install --upgrade pip

echo.
echo [3/3] 필요한 라이브러리 설치 중...
python -m pip install -r requirements.txt

echo.
echo ============================================
echo 설치가 완료되었습니다.
echo 이제 run.bat 파일을 실행하세요.
echo ============================================
echo.

pause