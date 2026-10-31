import hashlib
import json
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class Ledger:
    """Hash-chained local evidence, tamper-evident against unchanged external roots.

    This is not an authenticated audit log; a writer can regenerate the whole chain.
    """
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.previous='0'*64
        if self.path.exists():
            records=self.verify(self.path)
            if records:self.previous=records[-1]['hash']

    def append(self,event,payload):
        record={'previous':self.previous,'time_unix':time.time(),'event':event,'payload':payload}
        record['hash']=hashlib.sha256(canonical(record).encode()).hexdigest()
        with self.path.open('a',encoding='utf-8') as f:f.write(canonical(record)+'\n')
        self.previous=record['hash']
        return record

    @staticmethod
    def verify(path):
        previous='0'*64;rows=[]
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row=json.loads(line);digest=row.pop('hash')
            if row['previous']!=previous or hashlib.sha256(canonical(row).encode()).hexdigest()!=digest:
                raise ValueError('Evidence chain mismatch')
            row['hash']=digest;rows.append(row);previous=digest
        return rows
