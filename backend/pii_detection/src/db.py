from __future__ import annotations
import json
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

def get_engine() -> Engine:
    db_url = os.getenv("DB_URL", "mysql+pymysql://root:password@localhost:3306/ComplianceWizard")
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def insert_person_information(engine: Engine, *, source_id: str, source_path: str, text_value: str, entities_json: dict, model: str = "heuristic") -> int:
    sql = text("""
        INSERT INTO person_information (source_id, source_path, text, entities, model)
        VALUES (:source_id, :source_path, :text, :entities, :model)
    """)
    with engine.begin() as conn:
        res = conn.execute(sql, {
            "source_id": source_id,
            "source_path": source_path,
            "text": text_value,
            "entities": json.dumps(entities_json, ensure_ascii=False),
            "model": model,
        })
        return res.lastrowid if res else 0
