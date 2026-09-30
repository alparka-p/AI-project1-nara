#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
나라장터 자체입찰 공고 법령 위반사항 모니터링 AI
Final submission inference script

Required submit.zip layout:
    script.py
    model/judgment_item_catalog.json

Evaluation server provides:
    PPS_DATA_DIR
    PPS_OUTPUT_DIR
    PPS_MODEL_DIR
    PPS_EMBED_DIR

Local structural check:
    python script.py --data-dir ./open/data --input ./open/data/test.jsonl.gz --output-dir ./mock_output --mock
"""

from __future__ import annotations

import argparse
import csv
import gzip
import inspect
import json
import os
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

# ============================================================
# Official output contract
# ============================================================

ITEMS = [f"v{i}" for i in range(1, 25)]
EVIDENCE_COLS = [f"e{i}" for i in range(1, 25)]
OUTPUT_COLS = ["id"] + ITEMS + EVIDENCE_COLS
ABSENCE_ITEMS = {"v10", "v11", "v16", "v18", "v20"}

MAX_MODEL_LEN = 16384
MAX_INPUT_TOKENS = 13500
MAX_OUTPUT_TOKENS = 2200
MAX_EVIDENCE_CHARS = 500

# Local/mock char guard. Real server uses tokenizer hard check.
MOCK_PROMPT_CHAR_LIMIT = 26000

# ============================================================
# Retrieval patterns
# These are recall-oriented context retrieval hints, NOT labels.
# ============================================================

RAW_PATTERNS = {
    "qualification": [
        r"입찰\s*참가\s*자격", r"입찰참가자격", r"참가\s*자격",
        r"자격\s*요건", r"자격요건", r"참가자격",
    ],
    "performance": [
        r"실적", r"이행\s*실적", r"수행\s*실적", r"용역\s*실적",
        r"납품\s*실적", r"계약\s*실적",
    ],
    "region": [
        r"지역\s*제한", r"지역제한", r"본점\s*소재지", r"주된\s*영업소",
        r"사업장\s*소재지", r"소재지", r"관할\s*(?:시|군|구|도)", r"인접",
    ],
    "product_model": [
        r"모델\s*명", r"모델명", r"모델\s*번호", r"제품\s*번호", r"품\s*번",
        r"품번", r"제조\s*사", r"제조사", r"브랜드", r"상표",
    ],
    "direct_production": [
        r"직접\s*생산", r"직접생산", r"직접\s*생산\s*확인", r"직접생산확인",
    ],
    "enterprise": [
        r"중소기업자?", r"중기업", r"소기업", r"소상공인", r"중소\s*기업",
        r"소기업\s*[·ㆍ,\s및]*\s*소상공인",
    ],
    "supply_commitment": [
        r"물품\s*공급", r"물품공급", r"기술\s*지원", r"기술지원",
        r"확약서", r"협약서", r"공급\s*확인서", r"공급확인서",
    ],
    "software": [
        r"소프트웨어", r"\bSW\b", r"정보화\s*사업", r"정보화사업",
        r"대기업\s*참여", r"중소\s*소프트웨어",
    ],
    "joint_contract": [
        r"공동\s*수급", r"공동수급", r"공동\s*도급", r"공동도급",
        r"공동\s*이행", r"공동이행", r"분담\s*이행", r"분담이행",
        r"공동\s*계약", r"공동계약", r"출자\s*비율", r"출자비율",
        r"분담\s*비율", r"분담비율", r"지분율", r"100\s*분의\s*\d+",
        r"\d+(?:\.\d+)?\s*%",
    ],
    "briefing": [
        r"현장\s*설명", r"현장설명", r"제안요청\s*설명", r"제안\s*설명회", r"설명회",
    ],
    "deadline": [
        r"공고\s*기간", r"제안서\s*제출", r"입찰서\s*제출", r"제출\s*마감",
        r"마감\s*일", r"개찰", r"\d+\s*일",
    ],
    "contract_method": [
        r"일반\s*경쟁", r"제한\s*경쟁", r"지명\s*경쟁", r"수의\s*계약",
        r"협상에\s*의한\s*계약", r"협상\s*계약", r"적격\s*심사", r"최저가",
    ],
    "industry": [r"업종", r"면허", r"등록업체", r"등록\s*업체", r"허가", r"인가"],
    "amount": [
        r"추정\s*가격", r"추정가격", r"사업\s*예산", r"사업예산", r"배정\s*예산",
        r"예산\s*금액", r"예산액", r"사업\s*금액", r"사업금액", r"사업비",
        r"기초\s*금액", r"기초금액", r"예정\s*가격", r"예정가격",
    ],
}

PATTERNS = {k: [re.compile(p, re.I) for p in vals] for k, vals in RAW_PATTERNS.items()}

ITEM_CATEGORIES = {
    "v1": ["qualification"],
    "v2": ["performance", "qualification", "amount"],
    "v3": ["performance", "amount"],
    "v4": ["performance", "qualification", "amount"],
    "v5": ["region", "qualification", "amount"],
    "v6": ["region", "qualification", "amount"],
    "v7": ["region", "qualification", "amount"],
    "v8": ["performance", "region", "qualification"],
    "v9": ["product_model"],
    "v10": ["direct_production", "enterprise"],
    "v11": ["enterprise"],
    "v12": ["direct_production"],
    "v13": ["enterprise"],
    "v14": ["enterprise", "amount"],
    "v15": ["enterprise", "amount"],
    "v16": ["enterprise", "amount"],
    "v17": ["enterprise", "amount"],
    "v18": ["enterprise", "amount"],
    "v19": ["supply_commitment", "qualification"],
    "v20": ["software", "qualification", "amount"],
    "v21": ["joint_contract"],
    "v22": ["briefing", "qualification"],
    "v23": ["briefing", "deadline"],
    "v24": ["amount", "contract_method", "region", "industry"],
}

# Evidence-recovery patterns: used only after model predicts violation=1.
EVIDENCE_PATTERNS_RAW = {
    "v1": [r"참가\s*자격", r"입찰참가자격", r"기관", r"발주"],
    "v2": [r"실적", r"이행\s*실적", r"납품\s*실적", r"수행\s*실적"],
    "v3": [r"실적", r"배정\s*예산", r"예산", r"이상"],
    "v4": [r"실적", r"기관", r"발주", r"특정"],
    "v5": [r"지역\s*제한", r"소재지", r"본점", r"주된\s*영업소"],
    "v6": [r"소재지", r"본점", r"주된\s*영업소", r"시", r"군", r"구"],
    "v7": [r"인접", r"소재지", r"본점", r"지역"],
    "v8": [r"실적", r"지역", r"소재지", r"본점"],
    "v9": [r"모델\s*명", r"모델명", r"모델\s*번호", r"제품\s*번호", r"품번", r"제조\s*사", r"브랜드", r"상표"],
    "v12": [r"직접\s*생산", r"직접생산", r"직접\s*생산\s*확인", r"직접생산확인"],
    "v13": [r"소기업", r"소상공인", r"중소기업"],
    "v14": [r"중소기업", r"추정\s*가격", r"사업\s*금액"],
    "v15": [r"소기업", r"소상공인", r"중소기업"],
    "v17": [r"중소기업", r"소기업", r"소상공인"],
    "v19": [r"물품\s*공급", r"기술\s*지원", r"확약서", r"협약서", r"공급\s*확인서"],
    "v21": [r"공동\s*수급", r"공동\s*도급", r"출자\s*비율", r"분담\s*비율", r"지분율", r"5\s*%", r"10\s*%", r"100\s*분의\s*5", r"100\s*분의\s*10"],
    "v22": [r"현장\s*설명", r"설명회", r"참석", r"불참", r"입찰\s*참가"],
    "v23": [r"현장\s*설명", r"설명회", r"공고\s*기간", r"제안서\s*제출", r"마감", r"\d+\s*일"],
    "v24": [r"추정\s*가격", r"사업\s*예산", r"계약\s*방법", r"지역\s*제한", r"소재지", r"업종", r"면허"],
}

EVIDENCE_PATTERNS = {
    v: [re.compile(p, re.I) for p in EVIDENCE_PATTERNS_RAW.get(v, [])]
    for v in ITEMS
}

SYSTEM_PROMPT = """
너는 나라장터 자체입찰 공고의 24개 검토항목 판정 엔진이다.
반드시 제공된 대회 자료, 공고·첨부 원문, meta, 항목별 판단명세만 사용한다. 외부 지식을 추가하지 마라.

원칙:
- v1~v24를 모두 독립적으로 0/1로 판단한다.
- 검색 후보와 HIGH priority는 문맥 탐색 힌트일 뿐 정답이 아니다.
- 후보 문구가 없다는 이유만으로 비위반으로 판단하지 않는다.
- 공고문뿐 아니라 제공된 모든 첨부 문맥을 함께 고려한다.
- null과 "미입력"을 임의로 같은 의미로 처리하지 않는다.
- 세부품명 코드 조회가 불확실하거나 혼합이면 품목유형을 단정하지 않는다.
- v10,v11,v16,v18,v20은 부재탐지형이다. 완전관측이 아니거나 첨부가 탈락했으면 단순 문구 부재만으로 위반을 확정하지 않는다.
- v24는 법령 조문 위반이 아니라 공고/첨부 내용과 나라장터 meta의 실질적 불일치 여부다.
- evidence는 제공된 문서 원문에서 그대로 복사한 연속 부분문자열만 사용한다. 요약·수정·생성하지 않는다.
- v=0이면 evidence="". v10,v11,v16,v18,v20의 evidence는 항상 "".
- 설명문은 출력하지 말고 제공된 JSON schema에 맞는 객체만 출력한다.
""".strip()

# ============================================================
# I/O helpers
# ============================================================

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_schema(obj: dict) -> dict:
    def looks(x: Any) -> bool:
        return isinstance(x, dict) and any(
            k in x for k in ("type", "properties", "$schema", "$defs", "definitions", "oneOf", "anyOf", "allOf")
        )

    if looks(obj):
        return obj
    if looks(obj.get("schema")):
        return obj["schema"]
    x = obj.get("json_schema")
    if isinstance(x, dict):
        if looks(x):
            return x
        if looks(x.get("schema")):
            return x["schema"]
    raise RuntimeError(f"official JSON schema structure not recognized: {list(obj.keys())}")

# ============================================================
# Official SME competitive-product lookup
# ============================================================

def normalize_code(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) >= 8 else None


def read_csv_rows(path: Path) -> tuple[list[dict], list[str]]:
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows, list(reader.fieldnames or [])
        except Exception as e:
            last_error = e
    raise RuntimeError(f"cannot read SME CSV: {last_error}")


def build_product_index(path: Path) -> dict:
    rows, columns = read_csv_rows(path)

    # Prefer the exact official detail-product-number columns.
    exact_cols = [c for c in columns if "세부품명번호" in str(c)]
    fallback_cols = [c for c in columns if "품명번호" in str(c)]
    code_cols = exact_cols or fallback_cols
    exact_mode = bool(exact_cols)

    if not code_cols:
        raise RuntimeError(f"SME CSV product-code column not found: {columns}")

    codes: set[str] = set()
    lengths: dict[int, int] = {}
    for row in rows:
        for col in code_cols:
            code = normalize_code(row.get(col))
            if code:
                codes.add(code)
                lengths[len(code)] = lengths.get(len(code), 0) + 1

    if not codes:
        raise RuntimeError("SME CSV product-code index is empty")

    dominant_length = max(lengths, key=lengths.get)
    return {
        "codes": codes,
        "code_columns": code_cols,
        "exact_detail_columns": exact_mode,
        "dominant_length": dominant_length,
    }


def extract_product_codes(value: Any) -> list[str]:
    if value is None:
        return []
    text = " ".join(map(str, value)) if isinstance(value, list) else str(value)
    out: list[str] = []
    for raw in re.findall(r"(?<!\d)\d{8,12}(?!\d)", text):
        code = normalize_code(raw)
        if code and code not in out:
            out.append(code)
    return out


def product_lookup(codes: list[str], index: dict) -> dict:
    if not codes:
        return {
            "status": "UNKNOWN_NO_CODE",
            "matched_codes": [],
            "unmatched_codes": [],
            "hard_scope_safe": False,
        }

    expected_len = int(index["dominant_length"])
    valid = [c for c in codes if len(c) == expected_len]
    invalid_len = [c for c in codes if len(c) != expected_len]

    # Hard product gating is allowed only when we are using exact detail-product
    # columns and every code has the expected detail-code length.
    hard_scope_safe = bool(index["exact_detail_columns"] and valid and not invalid_len)

    matched = [c for c in valid if c in index["codes"]]
    unmatched = [c for c in valid if c not in index["codes"]]

    if invalid_len:
        status = "UNKNOWN_CODE_FORMAT"
    elif matched and unmatched:
        status = "MIXED"
    elif matched:
        status = "MATCH"
    elif unmatched:
        status = "NO_MATCH"
    else:
        status = "UNKNOWN_NO_CODE"

    return {
        "status": status,
        "matched_codes": matched,
        "unmatched_codes": unmatched,
        "invalid_length_codes": invalid_len,
        "expected_code_length": expected_len,
        "hard_scope_safe": hard_scope_safe,
    }


def product_scope(facts: dict) -> str:
    lookup = facts["product"]["lookup"]
    if not lookup.get("hard_scope_safe"):
        return "UNKNOWN"
    matched = lookup.get("matched_codes") or []
    unmatched = lookup.get("unmatched_codes") or []
    if matched and unmatched:
        return "MIXED"
    if matched:
        return "ALL_COMPETITIVE"
    if unmatched:
        return "ALL_GENERAL"
    return "UNKNOWN"

# ============================================================
# Candidate context extraction
# ============================================================

def line_spans(text: str) -> list[dict]:
    out: list[dict] = []
    offset = 0
    lines = text.splitlines(keepends=True) or ([text] if text else [])
    for i, line in enumerate(lines):
        start = offset
        end = start + len(line)
        out.append({"line_no": i + 1, "start": start, "end": end, "text": line})
        offset = end
    return out


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for start, end in windows[1:]:
        if start <= merged[-1][1] + 80:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def extract_blocks(doc: dict, patterns: list[re.Pattern], radius_lines: int = 1, max_blocks: int = 40) -> list[dict]:
    text = doc.get("text") or ""
    if not text:
        return []

    spans = line_spans(text)
    windows: list[tuple[int, int]] = []

    for i, span in enumerate(spans):
        line = span["text"]
        matches = []
        for p in patterns:
            matches.extend(list(p.finditer(line)))
        if not matches:
            continue

        # Normal line: keep neighboring lines for context.
        if len(line) <= 1600:
            left = max(0, i - radius_lines)
            right = min(len(spans) - 1, i + radius_lines)
            windows.append((spans[left]["start"], spans[right]["end"]))
        else:
            # Long reconstructed table: preserve a separate window around EACH
            # keyword occurrence instead of only the first match.
            for m in matches:
                absolute = span["start"] + m.start()
                start = max(span["start"], absolute - 420)
                end = min(span["end"], absolute + 900)
                windows.append((start, end))

    out: list[dict] = []
    for start, end in merge_windows(windows):
        raw = text[start:end]
        if not raw:
            continue
        matched_terms: list[str] = []
        for p in patterns:
            for m in p.finditer(raw):
                term = m.group(0)
                if term not in matched_terms:
                    matched_terms.append(term)
        out.append({
            "doc_id": doc.get("doc_id"),
            "doc_type": doc.get("type"),
            "start": start,
            "end": end,
            "matched_terms": matched_terms,
            "text": raw,
        })
        if len(out) >= max_blocks:
            break

    return out


def collect_categories(docs: list[dict]) -> dict[str, list[dict]]:
    return {
        category: [b for d in docs for b in extract_blocks(d, pats)]
        for category, pats in PATTERNS.items()
    }


def dedupe_blocks(blocks: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for b in blocks:
        key = (b["doc_id"], int(b["start"]), int(b["end"]))
        if key not in seen:
            seen.add(key)
            out.append(b)
    out.sort(key=lambda x: (str(x["doc_id"]), int(x["start"])))
    return out

# ============================================================
# Numeric/date extraction
# ============================================================

MONEY_RE = re.compile(
    r"(?<!\d)(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>억원|억|천만원|백만원|만원|천원|원)"
)
MONEY_UNIT = {
    "억원": 100_000_000,
    "억": 100_000_000,
    "천만원": 10_000_000,
    "백만원": 1_000_000,
    "만원": 10_000,
    "천원": 1_000,
    "원": 1,
}
AMOUNT_LABELS = ["추정가격", "사업예산", "배정예산", "예산금액", "예산액", "사업금액", "사업비", "기초금액", "예정가격"]

DATE_PATTERNS = [
    re.compile(r"(?P<y>20\d{2})[.\-/](?P<m>\d{1,2})[.\-/](?P<d>\d{1,2})"),
    re.compile(r"(?P<y>20\d{2})\s*년\s*(?P<m>\d{1,2})\s*월\s*(?P<d>\d{1,2})\s*일"),
]
PERCENT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*%")
HUNDREDTH_RE = re.compile(r"100\s*분의\s*(\d+(?:\.\d+)?)")


def extract_money_candidates(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        raw = block["text"]
        compacted = re.sub(r"\s+", "", raw)
        labels = [label for label in AMOUNT_LABELS if label in compacted]
        for m in MONEY_RE.finditer(raw):
            try:
                num = float(m.group("num").replace(",", ""))
                won = int(round(num * MONEY_UNIT[m.group("unit")]))
            except Exception:
                won = None
            result.append({
                "doc_id": block["doc_id"],
                "raw": m.group(0),
                "normalized_won": won,
                "labels": labels,
            })
    return result


def extract_date_candidates(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen = set()
    for block in blocks:
        raw = block["text"]
        for p in DATE_PATTERNS:
            for m in p.finditer(raw):
                try:
                    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
                    iso = f"{y:04d}-{mo:02d}-{d:02d}"
                except Exception:
                    iso = None
                key = (block["doc_id"], m.group(0), iso)
                if key not in seen:
                    seen.add(key)
                    result.append({"doc_id": block["doc_id"], "raw": m.group(0), "iso_date": iso})
    return result


def extract_percent_candidates(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        raw = block["text"]
        for m in PERCENT_RE.finditer(raw):
            result.append({"doc_id": block["doc_id"], "raw": m.group(0), "percent": float(m.group(1))})
        for m in HUNDREDTH_RE.finditer(raw):
            result.append({"doc_id": block["doc_id"], "raw": m.group(0), "percent": float(m.group(1))})
    return result


def unique_take(rows: list[dict], fields: list[str], limit: int) -> list[dict]:
    result: list[dict] = []
    seen = set()
    for row in rows:
        compact = {field: row.get(field) for field in fields}
        sig = json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        result.append(compact)
        if len(result) >= limit:
            break
    return result

# ============================================================
# Structured facts
# ============================================================

def build_facts(row: dict, product_index: dict) -> dict:
    meta = row.get("meta") or {}
    docs = row.get("docs") or []
    categories = collect_categories(docs)

    codes = extract_product_codes(meta.get("세부품명번호목록"))
    lookup = product_lookup(codes, product_index)

    core_meta = {k: meta.get(k) for k in [
        "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
        "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식",
        "정보화사업여부", "세부품명번호목록", "제한지역코드목록", "지역제한여부",
        "면허업종제한목록", "업종제한여부", "조항호내용", "공고게시일자",
        "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
    ]}

    completeness_raw = row.get("input_completeness") or {}
    completeness = {
        "fully_observed": bool(completeness_raw.get("완전관측", False)),
        "has_dropped_docs": bool(row.get("dropped_doc_counts") or {}),
        "dropped_doc_counts": row.get("dropped_doc_counts") or {},
    }

    money = extract_money_candidates(categories["amount"])
    dates = extract_date_candidates(categories["briefing"] + categories["deadline"])
    percents = extract_percent_candidates(categories["joint_contract"])

    high = [v for v in ITEMS if any(categories.get(c) for c in ITEM_CATEGORIES[v])]
    if lookup["status"] == "MATCH":
        high.extend(v for v in ("v10", "v11", "v13") if v not in high)
    elif lookup["status"] in {"NO_MATCH", "MIXED", "UNKNOWN_NO_CODE", "UNKNOWN_CODE_FORMAT"}:
        high.extend(v for v in ("v12", "v14", "v15", "v16", "v17", "v18") if v not in high)
    if "v24" not in high:
        high.append("v24")

    return {
        "meta": core_meta,
        "completeness": completeness,
        "product": {"codes": codes, "lookup": lookup},
        "candidates": categories,
        "money_candidates": money,
        "date_candidates": dates,
        "joint_percentages": percents,
        "high_priority_items": high,
    }

# ============================================================
# Rule sheet: preserve Stage-6 semantic fields
# ============================================================

def compact_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clip(value: Any, n: int) -> str:
    return compact_ws(value)[:n]


def compact_list(value: Any, item_limit: int = 3, char_limit: int = 180) -> str:
    if value is None:
        return ""
    if not isinstance(value, list):
        return clip(value, char_limit)
    parts = [compact_ws(x) for x in value if compact_ws(x)]
    return clip("; ".join(parts[:item_limit]), char_limit)


def legal_locator_summary(item_spec: dict, max_sources: int = 2) -> str:
    contexts = item_spec.get("legal_context") or []
    out: list[str] = []
    seen = set()
    for c in contexts:
        source = clip(c.get("source", ""), 52)
        locator = clip(c.get("locator", ""), 35)
        key = (source, locator)
        if key in seen:
            continue
        seen.add(key)
        text = (source + " " + locator).strip()
        if text:
            out.append(text)
        if len(out) >= max_sources:
            break
    return "; ".join(out)


def rule_sheet(catalog: dict, minimal: bool = False) -> str:
    lines: list[str] = []
    for v in ITEMS:
        x = catalog[v]
        tag = "ABS" if x.get("absence_detection") else "POS"
        amount = clip(x.get("amount_condition", ""), 135)
        exceptions = compact_list(x.get("exceptions", []), item_limit=2, char_limit=150)

        if minimal:
            parts = [
                f"{v} {x.get('name','')} [{tag}]",
                "위반:" + clip(x.get("violation_condition", ""), 150),
                "비위반:" + clip(x.get("nonviolation_condition", ""), 75),
            ]
            if amount:
                parts.append("금액:" + amount)
            if exceptions:
                parts.append("예외:" + exceptions)
            lines.append(" | ".join(parts))
            continue

        parts = [
            f"{v} {x.get('name','')} [{tag}]",
            "적용:" + clip(x.get("applicability", ""), 95),
            "위반:" + clip(x.get("violation_condition", ""), 190),
            "비위반:" + clip(x.get("nonviolation_condition", ""), 105),
        ]
        if amount:
            parts.append("금액:" + amount)
        if exceptions:
            parts.append("예외:" + exceptions)

        required_meta = compact_list(x.get("required_meta", []), item_limit=5, char_limit=100)
        facts = compact_list(x.get("facts_to_extract", []), item_limit=4, char_limit=120)
        if required_meta:
            parts.append("META:" + required_meta)
        if facts:
            parts.append("확인:" + facts)

        locator = legal_locator_summary(x)
        if locator:
            parts.append("근거:" + locator)

        lines.append(" | ".join(parts))

    return "\n".join(lines)

# ============================================================
# Prompt context selection
# ============================================================

def build_document_segments(row: dict, facts: dict, budget: int) -> list[dict]:
    docs = row.get("docs") or []
    all_blocks = dedupe_blocks([b for arr in facts["candidates"].values() for b in arr])
    by_doc: dict[str, list[dict]] = {}
    for b in all_blocks:
        by_doc.setdefault(b["doc_id"], []).append(b)

    selected: list[dict] = []
    seen = set()
    used = 0

    def add(seg: dict) -> None:
        nonlocal used
        key = (seg["doc_id"], int(seg["start"]), int(seg["end"]))
        if key in seen:
            return
        size = len(seg["text"])
        if used + size > budget:
            return
        # Skip a segment fully contained in an already selected one.
        for old in selected:
            if (
                old["doc_id"] == seg["doc_id"]
                and int(old["start"]) <= int(seg["start"])
                and int(old["end"]) >= int(seg["end"])
            ):
                return
        seen.add(key)
        selected.append(seg)
        used += size

    # Every document gets a small head so attachment-only violations remain visible.
    for d in docs:
        text = d.get("text") or ""
        head_len = min(450, len(text))
        if head_len:
            add({
                "doc_id": d["doc_id"], "doc_type": d.get("type"), "kind": "HEAD",
                "start": 0, "end": head_len, "text": text[:head_len],
            })

    # Round-robin candidate blocks across docs so one large notice cannot starve attachments.
    max_blocks = max((len(by_doc.get(d["doc_id"], [])) for d in docs), default=0)
    for k in range(max_blocks):
        for d in docs:
            arr = by_doc.get(d["doc_id"], [])
            if k < len(arr):
                add({**arr[k], "kind": "CANDIDATE"})

    # Small tail from each document.
    for d in docs:
        text = d.get("text") or ""
        if len(text) > 450:
            start = max(0, len(text) - 260)
            add({
                "doc_id": d["doc_id"], "doc_type": d.get("type"), "kind": "TAIL",
                "start": start, "end": len(text), "text": text[start:],
            })

    selected.sort(key=lambda x: (str(x["doc_id"]), int(x["start"])))
    return selected


def compact_facts(facts: dict) -> dict:
    lookup = facts["product"]["lookup"]
    return {
        "meta": facts["meta"],
        "completeness": facts["completeness"],
        "product": {
            "codes": facts["product"]["codes"],
            "status": lookup.get("status"),
            "matched_codes": lookup.get("matched_codes", []),
            "unmatched_codes": lookup.get("unmatched_codes", []),
            "hard_scope_safe": lookup.get("hard_scope_safe", False),
        },
        "money_candidates": unique_take(
            facts["money_candidates"], ["doc_id", "raw", "normalized_won", "labels"], 14
        ),
        "joint_percentages": unique_take(
            facts["joint_percentages"], ["doc_id", "raw", "percent"], 10
        ),
        "date_candidates": unique_take(
            facts["date_candidates"], ["doc_id", "raw", "iso_date"], 12
        ),
        "high_priority_items": facts["high_priority_items"],
    }


def make_prompt(row: dict, facts: dict, rules: str, doc_budget: int) -> dict:
    structured = json.dumps(compact_facts(facts), ensure_ascii=False, separators=(",", ":"), default=str)
    segs = build_document_segments(row, facts, doc_budget)
    docs_text = "\n\n".join(
        f"[D{i}] id={s['doc_id']} type={s.get('doc_type')} kind={s['kind']} offset={s['start']}:{s['end']}\n{s['text']}"
        for i, s in enumerate(segs, 1)
    )

    user = f"""[공고 ID]
{row['id']}

[구조화 사실]
{structured}

[24개 판정 명세]
{rules}

[공고 및 첨부 원문 문맥]
{docs_text}

[출력 지시]
공식 JSON schema에 맞춰 v1~v24를 모두 판단하라.
각 위반 여부는 정수 0 또는 1이다.
evidence는 위 원문에 실제 존재하는 연속 부분문자열만 사용한다.
v=0 및 v10,v11,v16,v18,v20의 evidence는 빈 문자열이다.""".strip()

    return {"system": SYSTEM_PROMPT, "user": user, "segments": segs}


def render_prompt(tokenizer: Any, prompt: dict) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def fit_prompt(tokenizer: Any, row: dict, facts: dict, full_rules: str, min_rules: str) -> tuple[str, int, str]:
    attempts = [
        (full_rules, "FULL", 9000),
        (full_rules, "FULL", 7000),
        (full_rules, "FULL", 5200),
        (full_rules, "FULL", 3800),
        (full_rules, "FULL", 2600),
        (min_rules, "MIN", 6000),
        (min_rules, "MIN", 4500),
        (min_rules, "MIN", 3200),
        (min_rules, "MIN", 2200),
    ]

    for rules, mode, budget in attempts:
        prompt = make_prompt(row, facts, rules, budget)
        rendered = render_prompt(tokenizer, prompt)
        tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if tokens <= MAX_INPUT_TOKENS:
            return rendered, tokens, mode

    raise RuntimeError(f"{row['id']}: prompt cannot fit {MAX_INPUT_TOKENS} input tokens")

# ============================================================
# Structured-output normalization
# ============================================================

ID_KEYS = {"item", "item_id", "violation_id", "key", "항목", "항목번호"}
VALUE_KEYS = {"violation", "violated", "value", "result", "label", "is_violation", "위반", "위반여부"}
EVIDENCE_KEYS = {"evidence", "evidence_text", "quote", "ground", "근거", "근거문구"}


def default_prediction() -> dict:
    result: dict[str, Any] = {}
    for i in range(1, 25):
        result[f"v{i}"] = 0
        result[f"e{i}"] = ""
    return result


def normalize_binary(value: Any) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "1.0", "true", "yes", "y"}:
            return 1
        if s in {"0", "0.0", "false", "no", "n", ""}:
            return 0
    try:
        x = int(value)
    except Exception:
        return None
    return x if x in (0, 1) else None


def normalize_item_id(value: Any) -> str | None:
    s = str(value).strip()
    if re.fullmatch(r"v(?:[1-9]|1\d|2[0-4])", s):
        return s
    if re.fullmatch(r"(?:[1-9]|1\d|2[0-4])", s):
        return "v" + s
    return None


def parse_json_output(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])
    raise ValueError("JSON parse failed")


def normalize_output(raw: Any) -> dict:
    obj = parse_json_output(raw)
    found_v: dict[str, int] = {}
    found_e: dict[str, str] = {}

    def put_v(item: Any, value: Any) -> None:
        item_id = normalize_item_id(item)
        binary = normalize_binary(value)
        if item_id and binary is not None:
            found_v[item_id] = binary

    def put_e(item: Any, value: Any) -> None:
        item_id = normalize_item_id(item)
        if item_id:
            found_e[f"e{int(item_id[1:])}"] = "" if value is None else str(value)

    def walk(x: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(x, dict):
            iid = None
            for key in ID_KEYS:
                if key in x:
                    candidate = normalize_item_id(x[key])
                    if candidate:
                        iid = candidate
                        break

            if iid:
                for key in VALUE_KEYS:
                    if key in x:
                        put_v(iid, x[key])
                        break
                for key in EVIDENCE_KEYS:
                    if key in x:
                        put_e(iid, x[key])
                        break

            for key, value in x.items():
                ks = str(key)
                if re.fullmatch(r"e(?:[1-9]|1\d|2[0-4])", ks) and not isinstance(value, (dict, list)):
                    found_e[ks] = "" if value is None else str(value)
                elif re.fullmatch(r"v(?:[1-9]|1\d|2[0-4])", ks):
                    if isinstance(value, dict):
                        for q in VALUE_KEYS:
                            if q in value:
                                put_v(ks, value[q])
                                break
                        for q in EVIDENCE_KEYS:
                            if q in value:
                                put_e(ks, value[q])
                                break
                    elif not isinstance(value, list):
                        evidence_container = any(
                            "evidence" in str(p).lower() or "근거" in str(p) for p in path
                        )
                        if evidence_container:
                            put_e(ks, value)
                        else:
                            put_v(ks, value)
                walk(value, path + (ks,))

        elif isinstance(x, list):
            for j, value in enumerate(x):
                walk(value, path + (str(j),))

    walk(obj)

    # Common 24-element array layouts.
    if isinstance(obj, dict):
        for key in ("violations", "labels", "results"):
            vals = obj.get(key)
            if isinstance(vals, list) and len(vals) == 24 and all(not isinstance(z, (dict, list)) for z in vals):
                for i, value in enumerate(vals, 1):
                    put_v(f"v{i}", value)
        for key in ("evidence", "evidences", "grounds"):
            vals = obj.get(key)
            if isinstance(vals, list) and len(vals) == 24 and all(not isinstance(z, (dict, list)) for z in vals):
                for i, value in enumerate(vals, 1):
                    put_e(f"v{i}", value)

    missing = [v for v in ITEMS if v not in found_v]
    if missing:
        raise ValueError(f"missing outputs: {missing}")

    result = default_prediction()
    for i in range(1, 25):
        v, e = f"v{i}", f"e{i}"
        result[v] = found_v[v]
        result[e] = "" if result[v] == 0 or v in ABSENCE_ITEMS else found_e.get(e, "")[:MAX_EVIDENCE_CHARS]
    return result

# ============================================================
# Safe deterministic postprocessing
# Only high-confidence 1 -> 0 corrections. No positive overrides.
# ============================================================

def safe_postprocess(prediction: dict, facts: dict) -> dict:
    result = deepcopy(prediction)
    scope = product_scope(facts)

    if scope == "ALL_GENERAL":
        for v in ("v10", "v11", "v13"):
            result[v] = 0
            result[f"e{int(v[1:])}"] = ""
    elif scope == "ALL_COMPETITIVE":
        for v in ("v12", "v14", "v15", "v16", "v17", "v18"):
            result[v] = 0
            result[f"e{int(v[1:])}"] = ""
    # MIXED / UNKNOWN -> preserve model judgment.

    for i in range(1, 25):
        v, e = f"v{i}", f"e{i}"
        result[v] = int(result.get(v, 0))
        if result[v] not in (0, 1):
            result[v] = 0
        if result[v] == 0 or v in ABSENCE_ITEMS:
            result[e] = ""
        else:
            result[e] = str(result.get(e, ""))[:MAX_EVIDENCE_CHARS]

    return result

# ============================================================
# Exact-substring evidence validator/recovery
# ============================================================

def find_exact(evidence: Any, row: dict) -> str | None:
    if evidence is None:
        return None
    evidence = str(evidence)
    if not evidence:
        return None
    for d in row.get("docs") or []:
        if evidence in (d.get("text") or ""):
            return evidence
    return None


def evidence_variants(evidence: Any) -> list[str]:
    if evidence is None:
        return []
    raw = str(evidence)
    result: list[str] = []

    def add(x: str) -> None:
        if x and x not in result:
            result.append(x)

    add(raw)
    stripped = raw.strip()
    add(stripped)
    if stripped.startswith("```") and stripped.endswith("```"):
        add(stripped[3:-3].strip())
    for left, right in [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")]:
        if len(stripped) >= 2 and stripped.startswith(left) and stripped.endswith(right):
            add(stripped[len(left):len(stripped) - len(right)])
    return result


def recover_whitespace(evidence: Any, row: dict) -> str | None:
    for variant in evidence_variants(evidence):
        target = "".join(ch for ch in variant if not ch.isspace())
        if len(target) < 6:
            continue
        for d in row.get("docs") or []:
            text = d.get("text") or ""
            chars: list[str] = []
            pos: list[int] = []
            for i, ch in enumerate(text):
                if not ch.isspace():
                    chars.append(ch)
                    pos.append(i)
            compact = "".join(chars)
            p = compact.find(target)
            if p >= 0:
                end = p + len(target) - 1
                if end < len(pos):
                    return text[pos[p]:pos[end] + 1]
    return None


def candidate_blocks_for_item(facts: dict, item: str) -> list[dict]:
    blocks: list[dict] = []
    seen = set()
    for category in ITEM_CATEGORIES[item]:
        for block in facts["candidates"].get(category, []):
            key = (block["doc_id"], int(block["start"]), int(block["end"]))
            if key not in seen:
                seen.add(key)
                blocks.append(block)
    return blocks


def evidence_block_score(block: dict, item: str) -> int:
    text = block["text"]
    score = sum(3 for p in EVIDENCE_PATTERNS[item] if p.search(text))
    score += min(len(block.get("matched_terms") or []), 4)
    if len(text) <= MAX_EVIDENCE_CHARS:
        score += 2

    if item == "v8" and re.search(r"실적", text) and re.search(r"지역|소재지|본점|영업소", text):
        score += 8
    if item == "v4" and re.search(r"실적", text) and re.search(r"기관|발주", text):
        score += 8
    if item == "v22" and re.search(r"설명", text) and re.search(r"참석|불참|참가", text):
        score += 8
    if item == "v23" and re.search(r"설명", text) and re.search(r"공고|기간|마감|제출|\d+\s*일", text):
        score += 8
    return score


def best_exact_span(block: dict, item: str) -> str | None:
    text = block["text"]
    patterns = EVIDENCE_PATTERNS[item]
    if not patterns:
        return None

    lines: list[tuple[int, int, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        score = sum(3 for p in patterns if p.search(stripped))
        if score > 0:
            lines.append((score, len(stripped), stripped))

    if lines:
        lines.sort(key=lambda x: (-x[0], x[1]))
        chosen = lines[0][2]
        if len(chosen) <= MAX_EVIDENCE_CHARS:
            return chosen
        for p in patterns:
            m = p.search(chosen)
            if m:
                start = max(0, m.start() - 150)
                end = min(len(chosen), start + MAX_EVIDENCE_CHARS)
                return chosen[start:end]

    for p in patterns:
        m = p.search(text)
        if m:
            start = max(0, m.start() - 150)
            end = min(len(text), start + MAX_EVIDENCE_CHARS)
            candidate = text[start:end].strip()
            return candidate or None
    return None


def recover_from_candidates(item: str, facts: dict, row: dict) -> str | None:
    scored = [(evidence_block_score(b, item), b) for b in candidate_blocks_for_item(facts, item)]
    scored.sort(key=lambda x: (-x[0], len(x[1]["text"])))

    for score, block in scored:
        # Require an item-specific keyword signal; generic category hit alone is insufficient.
        if score < 3:
            continue
        candidate = best_exact_span(block, item)
        if candidate and find_exact(candidate, row):
            return candidate
    return None


def validate_evidence(prediction: dict, row: dict, facts: dict) -> dict:
    result = deepcopy(prediction)

    for i in range(1, 25):
        v, e = f"v{i}", f"e{i}"
        if result[v] == 0 or v in ABSENCE_ITEMS:
            result[e] = ""
            continue

        raw = result.get(e, "")
        final = None

        for variant in evidence_variants(raw):
            exact = find_exact(variant, row)
            if exact:
                final = exact
                break

        if final is None:
            recovered = recover_whitespace(raw, row)
            if recovered and find_exact(recovered, row):
                final = recovered

        if final is None:
            final = recover_from_candidates(v, facts, row)

        if final:
            final = final[:MAX_EVIDENCE_CHARS]
            if not find_exact(final, row) or final.startswith(("=", "+", "@")):
                final = ""
        else:
            final = ""

        result[e] = final

    return result

# ============================================================
# vLLM structured output
# ============================================================

def build_sampling(schema: dict, max_tokens: int = MAX_OUTPUT_TOKENS):
    from vllm import SamplingParams

    supported = set(inspect.signature(SamplingParams).parameters)
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    if "seed" in supported:
        kwargs["seed"] = 42

    if "structured_outputs" in supported:
        try:
            from vllm.sampling_params import StructuredOutputsParams
            kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
        except ImportError:
            kwargs["structured_outputs"] = {"json": schema}
    elif "guided_decoding" in supported:
        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:
            from vllm import GuidedDecodingParams
        kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
    else:
        raise RuntimeError(f"vLLM structured-output API unsupported: {sorted(supported)}")

    return SamplingParams(**kwargs)


def create_llm(model_dir: Path):
    from vllm import LLM
    return LLM(
        model=str(model_dir),
        quantization="int8_per_channel_weight_only",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )

# ============================================================
# Submission validation/writing
# ============================================================

def make_output_row(row_id: Any, prediction: dict) -> dict:
    """Return one submission row in the exact official 49-column order."""
    out: dict[str, Any] = {"id": str(row_id)}
    for v in ITEMS:
        out[v] = int(prediction[v])
    for e in EVIDENCE_COLS:
        out[e] = str(prediction.get(e, "") or "")
    return out


def validate_output_rows(rows: list[dict], input_rows: list[dict]) -> None:
    input_ids = [str(r["id"]) for r in input_rows]
    if len(rows) != len(input_ids):
        raise RuntimeError(f"row count mismatch: {len(rows)} != {len(input_ids)}")

    ids = [str(r["id"]) for r in rows]
    if ids != input_ids:
        raise RuntimeError("output IDs/order do not match input")
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate output IDs")

    for out, source in zip(rows, input_rows):
        if list(out.keys()) != OUTPUT_COLS:
            raise RuntimeError(f"{out.get('id')}: output column order mismatch")
        for v in ITEMS:
            if out[v] not in (0, 1):
                raise RuntimeError(f"{out['id']} {v}: invalid value {out[v]}")
        for i in range(1, 25):
            v, e = f"v{i}", f"e{i}"
            ev = str(out[e] or "")
            if (out[v] == 0 or v in ABSENCE_ITEMS) and ev:
                raise RuntimeError(f"{out['id']} {e}: evidence must be empty")
            if len(ev) > MAX_EVIDENCE_CHARS:
                raise RuntimeError(f"{out['id']} {e}: evidence too long")
            if ev.startswith(("=", "+", "@")):
                raise RuntimeError(f"{out['id']} {e}: forbidden formula prefix")
            if ev and not find_exact(ev, source):
                raise RuntimeError(f"{out['id']} {e}: evidence is not an exact source substring")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLS,
            extrasaction="raise",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            normalized: dict[str, Any] = {}
            for col in OUTPUT_COLS:
                if col.startswith("v"):
                    normalized[col] = int(row[col])
                else:
                    normalized[col] = unicodedata.normalize("NFC", str(row[col] or ""))
            writer.writerow(normalized)

# ============================================================
# CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir")
    p.add_argument("--input")
    p.add_argument("--output-dir")
    p.add_argument("--catalog")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    data_dir = Path(args.data_dir or os.environ.get("PPS_DATA_DIR") or (root / "data"))
    input_path = Path(args.input) if args.input else (data_dir / "test.jsonl.gz")
    output_dir = Path(args.output_dir or os.environ.get("PPS_OUTPUT_DIR") or (root / "output"))
    catalog_path = Path(args.catalog) if args.catalog else (root / "model" / "judgment_item_catalog.json")
    schema_path = data_dir / "정답스키마_디코딩.json"
    sme_csv_path = data_dir / "법령패키지" / "중기부고시" / "중기부고시_경쟁제품_세부품명.csv"

    for path in (input_path, catalog_path, schema_path, sme_csv_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")

    rows = load_jsonl_gz(input_path)
    if not rows:
        raise RuntimeError("input test.jsonl.gz is empty")
    input_ids = [str(r["id"]) for r in rows]
    if len(input_ids) != len(set(input_ids)):
        raise RuntimeError("duplicate input IDs")

    catalog = load_json(catalog_path)
    if set(catalog) != set(ITEMS):
        raise RuntimeError("catalog must contain exactly v1..v24")

    # Core Stage-6 semantic fields must exist. Do not silently submit an old catalog.
    required_catalog_fields = {
        "name", "absence_detection", "applicability", "violation_condition",
        "nonviolation_condition", "amount_condition", "exceptions",
    }
    for v in ITEMS:
        missing = required_catalog_fields - set(catalog[v])
        if missing:
            raise RuntimeError(f"{v}: catalog missing required fields {sorted(missing)}")

    schema = resolve_schema(load_json(schema_path))
    product_index = build_product_index(sme_csv_path)
    full_rules = rule_sheet(catalog, minimal=False)
    min_rules = rule_sheet(catalog, minimal=True)

    facts_list = [build_facts(row, product_index) for row in rows]

    print("=" * 88)
    print("PPS final inference pipeline")
    print("=" * 88)
    print(f"records={len(rows)} mock={args.mock}")
    print(f"rule_chars={len(full_rules)}/{len(min_rules)}")
    print(f"SME_columns={product_index['code_columns']} exact_detail={product_index['exact_detail_columns']} code_len={product_index['dominant_length']}")

    # --------------------------------------------------------
    # MOCK: validate full preprocessing + prompt construction +
    # output contract without importing/loading vLLM.
    # --------------------------------------------------------
    if args.mock:
        max_chars = 0
        outputs: list[dict] = []
        for row, facts in zip(rows, facts_list):
            prompt = make_prompt(row, facts, full_rules, 9000)
            chars = len(prompt["system"]) + len(prompt["user"])
            if chars > MOCK_PROMPT_CHAR_LIMIT:
                prompt = make_prompt(row, facts, min_rules, 5000)
                chars = len(prompt["system"]) + len(prompt["user"])
            if chars > MOCK_PROMPT_CHAR_LIMIT:
                raise RuntimeError(f"{row['id']}: mock prompt too large: {chars}")
            max_chars = max(max_chars, chars)

            pred = safe_postprocess(default_prediction(), facts)
            pred = validate_evidence(pred, row, facts)
            outputs.append(make_output_row(row["id"], pred))

        validate_output_rows(outputs, rows)
        out_path = output_dir / "submission.csv"
        write_csv(outputs, out_path)
        print(f"mock_max_prompt_chars={max_chars}")
        print(f"[PASS] mock -> {out_path}")
        return

    # --------------------------------------------------------
    # REAL MODEL: LLM is created only inside main(), as required
    # by vLLM spawn behavior.
    # --------------------------------------------------------
    model_env = os.environ.get("PPS_MODEL_DIR")
    if not model_env:
        raise RuntimeError("PPS_MODEL_DIR missing")
    model_dir = Path(model_env)
    if not model_dir.exists():
        raise FileNotFoundError(f"PPS_MODEL_DIR not found: {model_dir}")

    llm = create_llm(model_dir)
    tokenizer = llm.get_tokenizer()
    sampling = build_sampling(schema)

    rendered: list[str] = []
    prompt_meta: list[tuple[int, str]] = []
    for i, (row, facts) in enumerate(zip(rows, facts_list), 1):
        text, token_count, mode = fit_prompt(tokenizer, row, facts, full_rules, min_rules)
        rendered.append(text)
        prompt_meta.append((token_count, mode))
        if i % 100 == 0 or i == len(rows):
            print(f"[prompt] {i}/{len(rows)} tokens={token_count} mode={mode}")

    outputs: list[dict] = []
    retry_queue: list[int] = []
    normal_response_seen: set[int] = set()
    batch_size = max(1, int(args.batch_size))

    # Primary pass: exactly one fixed-LLM request per notice.
    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        try:
            batch_result = llm.generate(rendered[start:end], sampling)
        except Exception as e:
            # Do not silently fabricate predictions if the model call itself failed.
            # Retry these notices individually below.
            print(f"[WARN] batch generate failure {start}:{end}: {type(e).__name__}: {e}", file=sys.stderr)
            retry_queue.extend(range(start, end))
            continue

        if len(batch_result) != end - start:
            print(f"[WARN] batch output count mismatch {start}:{end}", file=sys.stderr)
            retry_queue.extend(range(start, end))
            continue

        for j, model_out in enumerate(batch_result):
            idx = start + j
            if not model_out.outputs:
                retry_queue.append(idx)
                continue
            raw = model_out.outputs[0].text
            normal_response_seen.add(idx)
            try:
                pred = normalize_output(raw)
            except Exception as e:
                print(f"[WARN] parse failure id={rows[idx]['id']}: {e}", file=sys.stderr)
                retry_queue.append(idx)
                continue

            pred = safe_postprocess(pred, facts_list[idx])
            pred = validate_evidence(pred, rows[idx], facts_list[idx])
            outputs.append({"__index": idx, **make_output_row(rows[idx]["id"], pred)})

        print(f"[infer] {end}/{len(rows)}")

    # Retry only failed notices individually. A retry still uses the fixed LLM and
    # the notice content. This protects the whole submission from one malformed output.
    retry_queue = sorted(set(retry_queue))
    if retry_queue:
        print(f"[retry] notices={len(retry_queue)}")
        retry_sampling = build_sampling(schema, max_tokens=MAX_OUTPUT_TOKENS)

        for count, idx in enumerate(retry_queue, 1):
            success = False
            try:
                res = llm.generate([rendered[idx]], retry_sampling)
                if res and res[0].outputs:
                    normal_response_seen.add(idx)
                    raw = res[0].outputs[0].text
                    pred = normalize_output(raw)
                    pred = safe_postprocess(pred, facts_list[idx])
                    pred = validate_evidence(pred, rows[idx], facts_list[idx])
                    outputs.append({"__index": idx, **make_output_row(rows[idx]["id"], pred)})
                    success = True
            except Exception as e:
                print(f"[WARN] retry failure id={rows[idx]['id']}: {type(e).__name__}: {e}", file=sys.stderr)

            if not success:
                # Competition rule: every notice must receive a normal response from
                # the fixed model. A rule-only fallback is allowed here only if a
                # normal model response was received but parsing remained unusable.
                if idx not in normal_response_seen:
                    raise RuntimeError(
                        f"{rows[idx]['id']}: fixed-model call did not return a normal response "
                        "after retry; refusing non-compliant rule-only fallback"
                    )
                pred = safe_postprocess(default_prediction(), facts_list[idx])
                pred = validate_evidence(pred, rows[idx], facts_list[idx])
                outputs.append({"__index": idx, **make_output_row(rows[idx]["id"], pred)})

            if count % 20 == 0 or count == len(retry_queue):
                print(f"[retry] {count}/{len(retry_queue)}")

    # Sort by original test order and strip internal index.
    outputs.sort(key=lambda x: int(x["__index"]))
    final_rows: list[dict] = []
    for out in outputs:
        idx = int(out["__index"])
        pred = {k: out[k] for k in ITEMS + EVIDENCE_COLS}
        final_rows.append(make_output_row(rows[idx]["id"], pred))

    validate_output_rows(final_rows, rows)
    out_path = output_dir / "submission.csv"
    write_csv(final_rows, out_path)

    print("=" * 88)
    print("FINAL")
    print("=" * 88)
    print(f"rows={len(final_rows)}")
    print(f"retry_notices={len(retry_queue)}")
    print(f"output={out_path}")
    print("[PASS] submission.csv self-check")


if __name__ == "__main__":
    main()
