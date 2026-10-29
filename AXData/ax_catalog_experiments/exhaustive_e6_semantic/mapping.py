from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..core import file_hash
from .config import REFERENCE_FILE

TYPO_ALIASES = {"CC03_1_R_0": "CC03_1_R_O"}

DEVICE_RULES = (
    ("CC22_D", "dehumidifier", "제습"),
    ("CC22_H", "humidifier", "가습"),
    ("CC23_C", "cooling", "냉방"),
    ("CC23_H", "heating", "난방"),
    ("CC01", "roof_window", "천창"),
    ("CC03", "side_window", "측창"),
    ("CC04", "shade_curtain", "차광커튼"),
    ("CC05", "thermal_curtain", "보온커튼"),
    ("CC08", "circulation_fan", "순환팬"),
    ("CC09", "heater", "난방기"),
    ("CC10", "misting", "분무"),
    ("CC11", "grow_light", "보광"),
    ("CC12", "cooler", "냉방기"),
    ("CC13", "water_curtain", "순환식수막"),
    ("CC14", "co2_generator", "CO2"),
    ("CC17", "three_way_valve", "3WAY밸브"),
    ("CC18", "exhaust_fan", "배기팬"),
    ("CC19", "vent", "환기구"),
    ("CC20", "soil_irrigation", "토경 관수"),
    ("CC21", "hydroponic_pump", "수경 양수"),
    ("CC24", "fumigator", "훈증기"),
    ("CC25", "co2_heater", "CO2·난방 복합"),
    ("CC26", "irrigation_motor", "관수모터"),
    ("CC27", "irrigation_valve", "관수밸브"),
    ("CC28", "fertigation_motor", "관비모터"),
    ("CC29", "fertigation_valve", "관비밸브"),
    ("CC30", "flow_meter", "유량계"),
    ("CC31", "insulation_curtain", "단열커튼"),
    ("CC32", "insulation_curtain", "단열커튼"),
    ("CC33", "nutrient_irrigation_motor", "양액 관수모터"),
    ("CC34", "boiler", "보일러"),
)

PARENT_CODE = re.compile(r"^CC(?:\d{2})$")


def _device(code: str):
    for prefix, family, label in DEVICE_RULES:
        if code.startswith(prefix):
            return family, label
    return None, None


def _measurement(row):
    unit = str(row.unit or "").strip().upper()
    name = str(row.com_name or "")
    if unit == "Y/N":
        return "binary_status", "작동 상태", True
    if unit == "ON/OFF":
        return "binary_onoff", "작동 여부", True
    if unit in {"도", "℃", "°C"}:
        return "temperature", "온도", False
    if "개도" in name or "열림값" in name or "닫힘값" in name or "개수율" in name:
        return "opening_percent", "개도율", False
    if unit.startswith("TIME"):
        return "duration", "작동 시간", False
    return None, None, False


def build_mapping(reference: pd.DataFrame) -> pd.DataFrame:
    required = {"sect_code", "fld_code", "com_code", "com_name", "unit"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"sfkr_col_info columns missing: {sorted(missing)}")
    ref = reference.loc[reference.sect_code.astype(str).eq("CR")].copy()
    if ref.com_code.astype(str).duplicated().any():
        dup = sorted(ref.loc[ref.com_code.astype(str).duplicated(False), "com_code"].astype(str).unique())
        raise ValueError(f"Duplicate sfkr_col_info codes: {dup}")
    records = []
    for row in ref.itertuples(index=False):
        code = str(row.com_code).strip()
        family, family_kr = _device(code)
        measurement, measurement_kr, binary = _measurement(row)
        included = bool(family and measurement)
        if PARENT_CODE.fullmatch(code) or code in {"00", "CC"}:
            reason = "상위 분류 코드로 구체적인 측정 의미가 없음"
        elif not family:
            reason = "지원 장치군 규칙에 없음"
        elif not measurement:
            reason = "설정정보·횟수 등 v2 측정 유형 범위 밖"
        else:
            reason = ""
        records.append(
            {
                "source_code": code,
                "canonical_code": code,
                "com_name": str(row.com_name or "").strip(),
                "unit": str(row.unit or "").strip(),
                "device_family": family or "",
                "device_family_kr": family_kr or "",
                "measurement_type": measurement or "",
                "measurement_type_kr": measurement_kr or "",
                "is_binary": binary,
                "included": included,
                "mapping_action": "include" if included else "exclude",
                "mapping_reason": reason,
            }
        )
    by_code = {r["source_code"]: r for r in records}
    for alias, canonical in TYPO_ALIASES.items():
        if canonical not in by_code:
            raise ValueError(f"Typo alias target missing from sfkr_col_info: {canonical}")
        row = dict(by_code[canonical])
        row.update(
            source_code=alias,
            canonical_code=canonical,
            mapping_action="typo_alias",
            mapping_reason=f"숫자 0을 영문 O로 교정: {alias}→{canonical}",
        )
        records.append(row)
    result = pd.DataFrame(records).sort_values("source_code").reset_index(drop=True)
    included = result[result.included]
    conflicts = (
        included.groupby(["device_family", "measurement_type"])["unit"]
        .nunique()
        .loc[lambda x: x > 1]
    )
    if len(conflicts):
        raise ValueError(f"Semantic unit conflicts: {conflicts.to_dict()}")
    return result


def load_mapping(reference_file: Path = REFERENCE_FILE):
    reference_file = Path(reference_file)
    reference = pd.read_csv(reference_file, dtype=str, keep_default_na=False)
    return reference, build_mapping(reference), file_hash(reference_file)


def validate_observed_codes(codes, mapping: pd.DataFrame):
    known = set(mapping.source_code.astype(str))
    unknown = sorted(set(map(str, codes)) - known)
    if unknown:
        raise ValueError(f"Unknown E6 codes; update frozen mapping first: {unknown}")
