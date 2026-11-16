"""Small terminal formatters; safe to import before NumPy/CUDA initialization."""

import json


def timing_line(text, *, primary=False):
    style = '1;36' if primary else '36'
    return f'\033[{style}m{text}\033[0m'


_KIND_COLORS = dict(info='36', ok='32', warn='33', fail='31', dim='90')


def tag_line(tag, text, kind='info'):
    """`[tag] text`: tag in bold, both in the colour of the kind (info, ok, warn, fail, dim)."""
    color = _KIND_COLORS[kind]
    return f'\033[1;{color}m[{tag}]\033[0m \033[{color}m{text}\033[0m'


def log_tag(tag, text, kind='info'):
    print(tag_line(tag, text, kind), flush=True)


def task_result_text(result):
    status = str(result.get('status', ''))
    if result.get('success') is False:
        color, label = '31', '작업 실패'
    elif status.endswith('unverified') or (result.get('executed') is True and result.get('success') is not True):
        color, label = '33', '실행 완료 · 파지 성공 미확인'
    elif status == 'prepared':
        color, label = '32', '준비 완료'
    elif result.get('success') is True:
        color, label = '32', '작업 성공'
    elif status == 'planned':
        color, label = '36', '계획 완료'
    else:
        color, label = '33', '결과 확인 필요'
    body = json.dumps(result, ensure_ascii=False, indent=2)
    return f'\033[1;{color}m[Result] {label}\033[0m\n\033[{color}m{body}\033[0m'
