from queryGym.data.dataloader import UnifiedQuerySource
from pathlib import Path
import tempfile

def test_local_tsv_loading():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("1\tapple\n2\tbanana\n")
        path = f.name
    src = UnifiedQuerySource(backend="local", format="tsv", path=Path(path))
    items = list(src.iter())
    assert len(items) == 2 and items[0].qid == "1"
