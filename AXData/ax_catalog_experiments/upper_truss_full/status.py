import json
from .prepare import OUT


if __name__=='__main__':
    result={'root':str(OUT)}
    for name in ['supervisor','setup_verification']:
        p=OUT/f'{name}.json'
        result[name]=json.loads(p.read_text()) if p.exists() else None
    for mode in ['quick','formal']:
        result[mode]={}
        for name in ['pipeline','progress','publication','verification']:
            p=OUT/mode/f'{name}.json'
            result[mode][name]=json.loads(p.read_text()) if p.exists() else None
    print(json.dumps(result,ensure_ascii=False,indent=2))
