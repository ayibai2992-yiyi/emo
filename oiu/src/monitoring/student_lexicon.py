"""
校园/学生主题轻量词典先验。

用于 Layer1 快筛：命中考试、宿舍、就业、自伤隐喻等主题时提高送入深层的倾向。
不替代深度模型；`--force-deep` 评测路径不受影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# theme -> (keywords, risk_bonus)
STUDENT_THEME_LEXICON: Dict[str, Tuple[List[str], float]] = {
    "exam_academic": (
        [
            "挂科",
            "重修",
            "期末",
            "考试",
            "考研",
            "绩点",
            "GPA",
            "学业预警",
            "退学",
            "论文",
            "作业交不了",
            "听不懂课",
        ],
        1.2,
    ),
    "dorm_interpersonal": (
        [
            "宿舍",
            "室友",
            "舍友",
            "被排挤",
            "被孤立",
            "同学关系",
            "导师",
            "辅导员",
            "被霸凌",
            "被欺负",
        ],
        1.0,
    ),
    "career_future": (
        [
            "找不到工作",
            "就业",
            "实习",
            "秋招",
            "春招",
            "毕业即失业",
            "前途渺茫",
            "升学失败",
            "没学上",
        ],
        1.0,
    ),
    "family_pressure": (
        [
            "父母逼",
            "家里施压",
            "高考",
            "学费",
            "助学贷款",
            "嫌弃我",
            "对不起父母",
        ],
        1.1,
    ),
    "self_harm_metaphor": (
        [
            "不想上学了",
            "逃课一整天",
            "消失在学校",
            "从宿舍楼",
            "退学算了",
            "活着没意思",
            "撑不到毕业",
        ],
        2.0,
    ),
}


@dataclass
class StudentLexiconHit:
    themes: List[str]
    matched_keywords: List[str]
    risk_bonus: float
    theme_hit: bool


def match_student_lexicon(text: str) -> StudentLexiconHit:
    """扫描文本，返回主题命中与风险加成分（上限约 4.0）。"""
    if not text:
        return StudentLexiconHit([], [], 0.0, False)
    themes: List[str] = []
    matched: List[str] = []
    bonus = 0.0
    for theme, (kws, w) in STUDENT_THEME_LEXICON.items():
        hit_kw = [kw for kw in kws if kw in text]
        if hit_kw:
            themes.append(theme)
            matched.extend(hit_kw)
            bonus += w
    bonus = min(bonus, 4.0)
    return StudentLexiconHit(
        themes=themes,
        matched_keywords=matched,
        risk_bonus=float(bonus),
        theme_hit=bool(themes),
    )
