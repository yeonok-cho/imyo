#!/bin/bash
# roo_presentation 저장소를 초기화하고 발표 파일을 넣는 스크립트
# 사용법: bash deploy-to-roo-presentation.sh

set -e

REPO="https://github.com/yeonok-cho/roo_presentation.git"
TMPDIR=$(mktemp -d)

echo "📂 roo_presentation 클론 중..."
git clone "$REPO" "$TMPDIR/roo_presentation"
cd "$TMPDIR/roo_presentation"

echo "🗑️  기존 파일 삭제 중..."
git rm -rf . --quiet 2>/dev/null || true

echo "📋 발표 파일 복사 중..."
cp "$(dirname "$0")/roo-presentation.html" ./index.html
cp "$(dirname "$0")/PROMPT.md" ./PROMPT.md

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
