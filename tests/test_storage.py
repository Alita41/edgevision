from edgevision.types import Event
from edgevision.storage import EventStore


def ev(i,severity="observation"):
    return Event(i,"red sign-like candidate",.8,severity,i-1,i,(2,3,10,12),"heuristic", "test")


def test_sqlite_persistence_and_limit(tmp_path):
    db=tmp_path/"events.sqlite3"
    store=EventStore(db,limit=2)
    for i in range(1,4):store.add(ev(i, "notice" if i==3 else "observation"))
    assert store.count()==2
    assert [x["track_id"] for x in store.recent()]==[3,2]
    assert [x["track_id"] for x in store.recent(severity="notice")]==[3]
    store.close()
    again=EventStore(db)
    assert again.count()==2
    again.close()
