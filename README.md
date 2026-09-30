# NARA-AI

나라장터 입찰공고를 분석하여 사전에 정의된 24개 법령·규정 위반 항목을 자동 탐지하기 위한 AI 프로젝트입니다.

## Project Overview

본 프로젝트는 입찰공고의 자격 제한, 지역·실적 제한, 중소기업 관련 조건, 특수조건, 공고문 간 일관성 등을 분석하여 위반 가능성을 자동으로 판단하는 것을 목표로 합니다.

## Development Process

- Dev 데이터 구조 및 정답 분석
- 공식 평가 방식 재현
- Baseline 구축
- 24개 위반 항목별 판정 규칙 설계
- 법령 근거 기반 판단 로직 구성
- Structured Extraction
- 후처리 및 Consistency Rule 적용
- Mock Submission 검증
- 최종 제출 스크립트 구성

## Pipeline

```text
입찰공고
   ↓
데이터 전처리
   ↓
조건 및 핵심 정보 추출
   ↓
24개 항목별 판단
   ↓
법령 근거 및 규칙 적용
   ↓
후처리 / 일관성 검사
   ↓
최종 Submission 생성