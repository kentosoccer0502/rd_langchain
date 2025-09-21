import json
from pymongo import MongoClient
import msgpack

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "checkpointing_db"
THREAD_ID = "short_memory_example2"  # 必要に応じ変更

def try_unpack(b: bytes):
    # msgpack でデコードを試み、失敗したら utf-8 で表示
    try:
        return msgpack.unpackb(b, raw=False)
    except Exception:
        try:
            return b.decode("utf-8")
        except Exception:
            return repr(b)

def print_docs(coll_name, docs):
    print(f"=== {coll_name} ===")
    for d in docs:
        out = {k: v for k, v in d.items() if k != "value" and k != "checkpoint"}
        if "value" in d and d["value"] is not None:
            out["value_decoded"] = try_unpack(bytes(d["value"]))
        if "checkpoint" in d and d["checkpoint"] is not None:
            out["checkpoint_decoded"] = try_unpack(bytes(d["checkpoint"]))
        # metadata の Binary フィールドも展開
        meta = d.get("metadata")
        if isinstance(meta, dict):
            out["metadata_decoded"] = {}
            for k, v in meta.items():
                try:
                    out["metadata_decoded"][k] = v.decode("utf-8")
                except Exception:
                    out["metadata_decoded"][k] = v
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    wcoll = db.checkpoint_writes
    scoll = db.checkpoints

    writes = list(wcoll.find({"thread_id": THREAD_ID}).sort([("_id", -1)]).limit(20))
    snaps = list(scoll.find({"thread_id": THREAD_ID}).sort([("_id", -1)]).limit(10))

    print_docs("checkpoint_writes", writes)
    print_docs("checkpoints", snaps)

if __name__ == "__main__":
    main()