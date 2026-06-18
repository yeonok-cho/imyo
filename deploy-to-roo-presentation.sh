#!/bin/bash
# imyo(gh-pages)의 발표 자료를 roo_presentation 저장소로 복사하는 스크립트
#
# 사용법:
#   1) 이 파일을 본인 PC에 저장 (예: deploy-to-roo-presentation.sh)
#   2) 터미널에서 실행:
#        bash deploy-to-roo-presentation.sh
#
# 사전 준비:
#   - git 설치 필요
#   - GitHub 로그인 상태여야 함 (HTTPS면 자격 증명 캐시, SSH면 SSH 키 등록)
#   - yeonok-cho/imyo, yeonok-cho/roo_presentation 두 저장소에 push 권한 필요

set -e

SRC_REPO="https://github.com/yeonok-cho/imyo.git"
DST_REPO="https://github.com/yeonok-cho/roo_presentation.git"
TMPDIR=$(mktemp -d)

echo "📂 imyo (gh-pages) 클론 중..."
git clone --branch gh-pages --single-branch "$SRC_REPO" "$TMPDIR/imyo"

echo "📂 roo_presentation 클론 중..."
git clone "$DST_REPO" "$TMPDIR/roo_presentation"
cd "$TMPDIR/roo_presentation"

echo "🗑️  기존 파일 삭제 중..."
git rm -rf . --quiet 2>/dev/null || true

echo "📋 발표 파일 복사 중..."
cp "$TMPDIR/imyo/index.html" ./index.html
cp "$TMPDIR/imyo/PROMPT.md" ./PROMPT.md

echo "💾 커밋 중..."
git add .
git commit -m "Roo AI 코딩 에이전트 소개 발표 자료 + 재현 프롬프트"

echo "🚀 푸시 중..."
git push origin main

echo ""
echo "✅ 완료! https://github.com/yeonok-cho/roo_presentation"
echo ""
echo "GitHub Pages 활성화가 안 되어 있다면:"
echo "  Settings → Pages → Source: Deploy from branch → main / (root) → Save"
echo "  활성화 후 URL: https://yeonok-cho.github.io/roo_presentation/"

cd ~
rm -rf "$TMPDIR"
