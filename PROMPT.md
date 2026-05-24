# HTML 프레젠테이션 제작 프롬프트

이 파일을 Claude에게 붙여넣으면 **동일한 디자인·구조·인터랙션**으로 새로운 발표 자료를 만들 수 있습니다.  
`[대괄호]` 부분만 바꾸면 됩니다.

---

## 바로 쓰는 프롬프트 (복사해서 Claude에게 붙여넣기)

```
아래 스펙에 맞춰 HTML 프레젠테이션을 단일 파일로 만들어줘.

## 발표 정보
- 제목: [발표 제목]
- 부제: [한 줄 부제]
- 발표자: [이름 / 소속]
- 슬라이드 수: [총 몇 장]
- 언어: 한국어

## 슬라이드 내용
[아래처럼 각 슬라이드 내용을 붙여넣거나 설명해줘]

1페이지 - 타이틀
- 메인 제목: [제목]
- 첫 문장 클릭하면 두 번째 문장이 나타나는 인터랙션
- 하단에 각 슬라이드로 이동하는 키워드 카드 (클릭 시 해당 슬라이드로 점프)

2페이지 - [소주제]
- [내용]

3페이지 - [소주제]
- [내용]

...

## 디자인 요구사항
아래 디자인 시스템을 그대로 사용해줘.

### 색상 토큰 (CSS 변수)
--bg:        #FAFAF8   (전체 배경)
--surface:   #FFFFFF   (카드·패널 배경)
--border:    #E8E4DE   (테두리)
--border-sm: #F0EDE8   (얇은 테두리)
--primary:   #D97757   (포인트 컬러 — 오렌지)
--primary-h: #C4623E   (포인트 hover)
--primary-bg:#FDF3EE   (포인트 연한 배경)
--primary-br:#F9D5C3   (포인트 테두리)
--text:      #1C1917   (본문 텍스트)
--muted:     #78716C   (보조 텍스트)
--soft:      #F5F2EE   (hover 배경)
--green:     #16A34A / --green-bg: #DCFCE7 / --green-br: #BBF7D0
--red:       #DC2626 / --red-bg:  #FEE2E2
--blue:      #2563EB / --blue-bg: #EFF6FF / --blue-br: #BFDBFE
--radius:    12px
--radius-sm: 8px
--shadow:    0 1px 4px rgba(0,0,0,.07), 0 4px 16px rgba(0,0,0,.05)

### 레이아웃 구조
- 전체: flex column, height: 100dvh, overflow: hidden (PPT처럼 한 페이지가 화면을 꽉 채움)
- 상단바: 브랜드명 + 진행 프로그레스 바 + 이전/다음 버튼
- 슬라이드 영역: flex 1, overflow hidden, 슬라이드는 translateX로 전환
- 하단바: 점(dot) 페이지 인디케이터
- 모든 슬라이드: overflow: hidden (스크롤 없음, PPT 고정 높이)

### 반드시 포함할 기능
1. 키보드 방향키 (←→) 슬라이드 전환
2. 터치 스와이프 (모바일 대응)
3. 프로그레스 바 + 슬라이드 번호 표시
4. 🐾 발바닥 홈 버튼: 짝수 페이지(2,4,6,8...)에만 표시, 클릭 시 1페이지로 이동
   - 평소 opacity 0.28 + 살짝 위아래 bounce 애니메이션
   - hover 시 opacity 0.8로 강조
5. Escape 키로 전체화면 오버레이 닫기

### 인터랙티브 컴포넌트 (필요한 것만 골라 사용)

**클릭 reveal (1페이지용)**
- 첫 번째 문장 클릭 → 두 번째 문장 max-height 0→60px 애니메이션으로 나타남
- 동시에 발표자 정보 translateY(8px)→0 + opacity 0→1

**모드 카드 (접었다 펴기)**
- flex 컨테이너 고정 높이 (height: 96px)
- 클릭한 카드: flex 2.8로 늘어남 + 설명 텍스트 나타남
- 나머지 카드: flex 0 0 58px로 줄어들어 아이콘만 보임
- 컨테이너 전체 높이는 변하지 않음

**전체화면 오버레이**
- position: fixed; inset: 0; z-index: 400
- 평소: translateY(100%)로 화면 아래 숨김
- 열릴 때: translateY(0)로 올라옴 (transition 0.35s)
- 내부 스크롤 가능 (overflow-y: auto)
- 닫기 버튼 또는 Escape로 닫힘

**비교 테이블**
- 첫 번째 열(항목): width 80px, font-weight 600, color var(--muted)
- Roo/핵심 컬럼: background #FFFBF9, font-weight 500
- 헤더: background var(--soft), uppercase, letter-spacing

### 폰트
- 시스템 폰트: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif
- 슬라이드 제목: clamp(1.5rem, 3.5vw, 2.2rem), font-weight 800, letter-spacing -.04em
- 본문: 0.8~0.95rem

### 다크 모드 절대 사용하지 말 것 (라이트 모드 전용)

## 출력 형식
- 단일 HTML 파일 (CSS·JS 모두 인라인)
- 외부 라이브러리 없음 (vanilla CSS·JS만)
- 파일명: [원하는 파일명].html
```

---

## 슬라이드 구성 팁

### 슬라이드 당 적정 내용량
| 요소 | 권장 |
|---|---|
| 제목 | 15자 이내 |
| 부제 | 30자 이내 |
| 카드/항목 수 | 3~4개 |
| 텍스트 줄 수 | 카드당 2~3줄 |

전체 슬라이드가 동일한 세로 길이를 유지해야 합니다.  
내용이 많은 슬라이드는 **모드 카드(접기/펼치기)** 또는 **전체화면 오버레이**로 처리하세요.

### 슬라이드 흐름 추천 구조 (10장 기준)
```
1페이지  타이틀 + 키워드 네비게이션 카드
──────────────── 챕터 1 ────────────────
2페이지  개념 설명 A         ← 🐾 발바닥 (챕터 종료 신호)
3페이지  개념 설명 B (심화)
──────────────── 챕터 2 ────────────────
4페이지  문제 제기 / 한계     ← 🐾 발바닥
5페이지  해결책 / 소개
──────────────── 챕터 3 ────────────────
6페이지  비교 / 선택 기준     ← 🐾 발바닥
7페이지  실습 / 설치 방법
──────────────── 챕터 4 ────────────────
8페이지  사용법 / 팁          ← 🐾 발바닥
9페이지  실제 사례 / 시나리오
──────────────── 마무리 ────────────────
10페이지 정리 + Q&A           ← 🐾 발바닥
```

발바닥 버튼이 나타나면 → 1페이지로 돌아가 → 다음 키워드 카드 클릭 → 다음 챕터로 이동

---

## 컴포넌트 스니펫

### 키워드 네비게이션 카드 (1페이지용)
```html
<div class="tag-row">
  <div class="tag-item" onclick="goto(1)">
    <span class="tag-name">키워드</span>
    <span class="tag-desc">한 줄 설명</span>
  </div>
  <!-- 슬라이드 수만큼 반복 -->
</div>
```

### 모드 카드 (접기/펼치기)
```html
<div class="mode-grid">
  <div class="mode-card" onclick="toggleMode(this)">
    <div class="mode-card-header">
      <span class="mode-icon">🛠️</span>
      <span class="mode-label">모드명</span>
    </div>
    <div class="mode-card-detail">설명 텍스트</div>
  </div>
</div>
```

### 전체화면 오버레이 (팁/상세 내용용)
```html
<!-- 트리거 버튼 -->
<button onclick="openTips()">💡 실전 팁 보기</button>

<!-- 오버레이 -->
<div class="tips-overlay" id="tipsOverlay">
  <div class="tips-header">
    <span>실전 팁</span>
    <button onclick="closeTips()">✕ 닫기</button>
  </div>
  <div class="tips-body">
    <!-- 내용 (스크롤 가능) -->
  </div>
</div>
```

### 비교 테이블
```html
<div class="tbl-wrap">
  <table class="cmp-table">
    <thead>
      <tr>
        <th></th>
        <th>옵션 A</th>
        <th class="roo-col">★ 추천</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space:normal;">항목<br>이름</td>
        <td>내용</td>
        <td class="roo-col">내용</td>
      </tr>
    </tbody>
  </table>
</div>
```

---

## 파일 구조

```
your-presentation.html   ← 단일 파일 (CSS·JS 모두 포함)
PROMPT.md                ← 이 파일 (재현용 프롬프트)
```

GitHub Pages 배포:
1. 저장소 Settings → Pages → Source: Deploy from a branch
2. Branch: `gh-pages` (또는 `main`), folder: `/ (root)`
3. `index.html`로 파일명 저장 후 push

---

## 예시: 이 발표 자료 정보

| 항목 | 내용 |
|---|---|
| 제목 | Roo — AI와 함께 일하는 새로운 방법 |
| 대상 | AI 도구를 처음 접하는 직장인 |
| 슬라이드 수 | 10장 |
| 라이브 URL | https://yeonok-cho.github.io/imyo/ |
| 주요 인터랙션 | 클릭 reveal, 모드 카드 flex 펼치기, 실전 팁 전체화면 오버레이, 🐾 챕터 홈 버튼 |
