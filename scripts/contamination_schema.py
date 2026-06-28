import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class ContaminationRecord:
    qid: str
    question: str
    answer: str
    model_id: str
    is_contam: int
    correct: int
    correct_clean: Optional[int] = None
    response: Optional[str] = None
    source_task: Optional[str] = None
    source_split: Optional[str] = None
    source_index: Optional[int] = None
    contam_type: Optional[str] = None
    contam_prompt: Optional[str] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "contaminated"}:
        return 1
    if lowered in {"0", "false", "no", "clean", "uncontaminated"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pick(row: Dict, *keys, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def convert_legacy_row(
    row: Dict,
    model_id: str,
    source_task: Optional[str] = None,
    source_split: Optional[str] = None,
    contam_type: str = "fewshot",
) -> ContaminationRecord:
    qid = str(_pick(row, "qid", "doc_id", "question_id"))
    question = str(_pick(row, "question", "input"))
    answer = str(_pick(row, "answer", "target"))
    is_contam = normalize_label(_pick(row, "is_contam", default=0))
    correct = normalize_label(_pick(row, "correct", "acc_norm", default=0))
    correct_clean = _pick(row, "correct_clean", "acc_norm")
    if correct_clean is not None:
        correct_clean = normalize_label(correct_clean)
    source_index = _pick(row, "source_index", "doc_id")
    if source_index is not None:
        source_index = int(source_index)
    return ContaminationRecord(
        qid=qid,
        question=question,
        answer=answer,
        model_id=model_id,
        is_contam=is_contam,
        correct=correct,
        correct_clean=correct_clean,
        response=_pick(row, "response"),
        source_task=source_task or _pick(row, "source_task"),
        source_split=source_split or _pick(row, "source_split"),
        source_index=source_index,
        contam_type=_pick(row, "contam_type", default=contam_type),
        contam_prompt=_pick(row, "contam_prompt"),
    )


def convert_legacy_rows(
    rows: Iterable[Dict],
    model_id: str,
    source_task: Optional[str] = None,
    source_split: Optional[str] = None,
    contam_type: str = "fewshot",
) -> List[Dict]:
    return [
        convert_legacy_row(
            row,
            model_id=model_id,
            source_task=source_task,
            source_split=source_split,
            contam_type=contam_type,
        ).to_dict()
        for row in rows
    ]


def infer_task_name(path: str) -> str:
    return Path(path).stem
