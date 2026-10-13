"""Export the local structured event audit trail without exposing camera frames."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from edgevision.storage import EventStore


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",default="data/events.sqlite3")
    ap.add_argument("--out",default="artifacts/event-export.jsonl")
    ap.add_argument("--limit",type=int,default=200)
    args=ap.parse_args()
    if not Path(args.db).exists():raise SystemExit(f"event database does not exist: {args.db}")
    store=EventStore(args.db)
    result=store.recent(min(max(args.limit,1),200))
    store.close()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in reversed(result)),encoding="utf-8")
    print(f"Exported {len(result)} event metadata records to {path}")

if __name__=="__main__":main()
