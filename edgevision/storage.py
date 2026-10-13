"""Local metadata-only event storage. Camera frames are never written here."""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from .types import Event


class EventStore:
    def __init__(self,path:str | Path=":memory:",limit:int=10000):
        self.path=str(path)
        if self.path!=":memory:":
            Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.path,check_same_thread=False)
        self.conn.row_factory=sqlite3.Row
        self.lock=threading.RLock()
        self.limit=limit
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_time REAL NOT NULL,
                  track_id INTEGER NOT NULL,
                  label TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  severity TEXT NOT NULL,
                  source TEXT NOT NULL,
                  model_version TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time DESC);
                CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
            """)
            self.conn.commit()

    def add(self,event:Event)->int:
        obj=event.public()
        with self.lock:
            cur=self.conn.execute("INSERT INTO events (event_time,track_id,label,confidence,severity,source,model_version,payload) VALUES (?,?,?,?,?,?,?,?)",(
                event.confirmed_at,event.track_id,event.label,event.confidence,event.severity,event.source,event.model_version,json.dumps(obj,ensure_ascii=False)))
            self.conn.execute("DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id DESC LIMIT -1 OFFSET ?)",(self.limit,))
            self.conn.commit()
            return int(cur.lastrowid)

    def recent(self,limit:int=30,severity:str|None=None)->list[dict]:
        limit=min(max(int(limit),1),200)
        with self.lock:
            if severity:
                rows=self.conn.execute("SELECT id,payload FROM events WHERE severity=? ORDER BY event_time DESC,id DESC LIMIT ?",(severity,limit)).fetchall()
            else:
                rows=self.conn.execute("SELECT id,payload FROM events ORDER BY event_time DESC,id DESC LIMIT ?",(limit,)).fetchall()
        return [{"id":row["id"],**json.loads(row["payload"])} for row in rows]

    def count(self)->int:
        with self.lock:return int(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self):
        with self.lock:self.conn.close()
