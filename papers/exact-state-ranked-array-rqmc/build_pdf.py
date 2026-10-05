"""Typeset the Paper I working manuscript; never execute a study."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
WIDTHS = {'1': (16, 26, 33, 25), '2': (10, 8, 12, 14, 14, 14, 14, 14),
          '3': (30, 14, 14, 14, 14, 14), '4': (21, 8, 11, 21, 25, 14),
          '5': (25, 10, 13, 12, 14, 13, 13), 'C1': (28, 16, 14, 14, 14, 14)}
SUPPLEMENT_WIDTHS = {'S1': (5, 14, 16, 16, 16, 33)}
SUPPLEMENT_TABLE_REFS = {'4': 'S1'}
sys.path.insert(0, str(ROOT/'.pdf-tools'))
import pypandoc
import typst


def plain(nodes):
    text = []
    for n in nodes:
        if n['t'] == 'Str':
            text.append(n['c'])
        elif n['t'] in ('Space', 'SoftBreak', 'LineBreak'):
            text.append(' ')
        elif n['t'] in ('Strong','Emph'):
            text.append(plain(n['c']))
    return ''.join(text)


def maths(x):
    if isinstance(x, list):
        return sum((maths(y) for y in x), [])
    if isinstance(x, dict):
        return [x['c']] if x.get('t') == 'Math' else maths(x.get('c', []))
    return []


def raw(s):
    return dict(t='RawBlock', c=['typst',s])


def layout(blocks, table_widths):
    out = []
    figures = tables = 0
    table_ids = set()
    keep_depth = 0
    i = 0
    while i < len(blocks):
        b = deepcopy(blocks[i])
        if b['t'] == 'RawBlock' and b['c'][0] == 'html':
            marker = b['c'][1].strip()
            if marker in ('<!-- keep-begin -->', '<!-- keep-end -->'):
                if marker == '<!-- keep-begin -->':
                    assert keep_depth == 0
                    keep_depth = 1
                    out.append(raw('#keep['))
                else:
                    assert keep_depth == 1
                    keep_depth = 0
                    out.append(raw(']'))
                i += 1
                continue
        if b['t'] == 'Header':
            b['c'][0] -= 1
            if plain(b['c'][2]) == 'Author information':
                assert i+1 < len(blocks) and blocks[i+1]['t'] == 'Para'
                # Treat the contact details as compact end matter at the
                # reference type size, rather than a new full-size section.
                out.extend([raw('#keep[\n#set text(size: 9pt)\n#set par(justify: false)\n#strong[Author information:]\n'),
                            deepcopy(blocks[i+1]), raw(']')])
                i += 2
                continue
            if plain(b['c'][2]) == 'References':
                out.extend([b,
                            raw('#set text(size: 9pt)\n#set par(leading: .45em, spacing: .45em)')])
                i += 1
                continue
        if b['t'] == 'Para' and any(n['t']=='Image' for n in b['c']):
            caption = deepcopy(blocks[i+1])
            assert caption['t']=='Para' and plain(caption['c']).startswith('Figure ')
            out.extend([raw('#keep['),b,raw('#set text(size: 9pt)\n#set par(justify: false)'),caption,raw(']')])
            figures += 1
            i += 2
            continue
        if (b['t'] == 'Para' and re.match(r'^Algorithm [A-Z]?\d+\.', plain(b['c']))
                and i+1 < len(blocks) and blocks[i+1]['t'] == 'CodeBlock'):
            out.extend([raw('#keep['), b, deepcopy(blocks[i+1]), raw(']')])
            i += 2
            continue
        if b['t']=='Table':
            raise ValueError('Every table must have its numbered caption immediately before it')
        elif b['t']=='Para' and i+1 < len(blocks) and blocks[i+1]['t']=='Table':
            # Caption with the table, using a smaller type only within this block.
            table = deepcopy(blocks[i+1])
            match = re.match(r'^Table ([A-Z]?\d+)\.', plain(b['c']))
            assert match, 'Missing numbered table caption'
            table_id = match.group(1)
            assert table_id not in table_ids
            table_ids.add(table_id)
            widths = table_widths[table_id]
            assert len(table['c'][2])==len(widths)
            for col,width in zip(table['c'][2],widths):
                col[0] = dict(t='AlignLeft')
                col[1] = dict(t='ColWidth',c=width/sum(widths))
            out.extend([raw('#keep['),raw('#set text(size: 9pt)\n#set par(justify: false)'),b,table,raw(']')])
            tables += 1
            i += 1
        elif b['t'] == 'Para' and re.match(r'^(?:Theorem|Proposition|Lemma|Corollary) [A-Z]?\d+ ', plain(b['c'])):
            out.extend([raw('#block(sticky: true)['), b, raw(']')])
        elif (b['t'] == 'Para' and i+1 < len(blocks)
              and blocks[i+1]['t'] == 'Para'
              and len(blocks[i+1]['c']) == 1
              and blocks[i+1]['c'][0]['t'] == 'Math'
              and blocks[i+1]['c'][0]['c'][0] == 'DisplayMath'):
            # Keep an equation's introduction with its display.
            out.extend([raw('#block(sticky: true)['), b, raw(']')])
        else:
            out.append(b)
        i += 1
    assert table_ids==set(table_widths)
    assert keep_depth == 0
    return out, figures, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement', action='store_true',
                        help='Build the separately numbered retained studies.')
    args = parser.parse_args()
    stem = 'retained-studies' if args.supplement else 'implicit-array-rqmc'
    source_name = 'retained_studies.md' if args.supplement else 'main.md'
    source = (ROOT/source_name).read_text(encoding='utf-8')
    # Numerical tables are regenerated from the retained raw observations.
    def include(match):
        included = (ROOT/match.group(1)).read_text(encoding='utf-8')
        if match.group(2):
            included, count = re.subn(r'(?m)^(\*\*Table )[A-Z]?\d+\.',
                                     lambda m: m[1] + match.group(2) + '.', included)
            assert count == 1
        if args.supplement:
            included = re.sub(r'\bTable ([A-Z]?\d+)\b',
                lambda m: 'Table ' + SUPPLEMENT_TABLE_REFS.get(m[1], m[1]), included)
        return included
    source = re.sub(r'<!-- include: ([^;\n]+)(?:; table: ([A-Z]?\d+))? -->', include, source)
    ast = json.loads(pypandoc.convert_text(source,'json',format='markdown+tex_math_dollars-implicit_figures'))
    title = plain(ast['blocks'][0]['c'][2])
    before = maths(ast['blocks'])
    ast['blocks'], figures, tables = layout(ast['blocks'][1:],
        SUPPLEMENT_WIDTHS if args.supplement else WIDTHS)
    assert before == maths(ast['blocks'])
    body = pypandoc.convert_text(json.dumps(ast), 'typst', format='json', extra_args=['--wrap=none'])
    # Pandoc emits CRLF on Windows. A second text-mode conversion would
    # produce CR-CR-LF and add blank lines inside raw algorithm blocks.
    body = body.replace('\r\n', '\n').replace('\r', '\n')
    body = re.sub(r'image\("(figures/[^\"]+)"\)', r'image("/\1", width: 100%)', body)
    body = body.replace('#h(-1em)', '#h(-0.166667em)')
    running_title = 'Array-RQMC: supplementary material' if args.supplement else 'Exact Array-RQMC: representations and shared work'
    preamble = ('#let manuscript-title = '+json.dumps(title)+'\n'
                '#let manuscript-running-title = '+json.dumps(running_title)+'\n'
                '#let manuscript-equation-numbering = '+json.dumps('(S1)' if args.supplement else '(1)')+'\n')
    generated = ROOT/('tmp/pdfs/'+stem+'.typ')
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(preamble+(ROOT/'pdf-style.typ').read_text(encoding='utf-8')+'\n'+body,encoding='utf-8',newline='\n')
    output = ROOT/('output/pdf/'+stem+'.pdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    _, warnings = typst.compile_with_warnings(generated,output=output,root=ROOT,
                ignore_system_fonts=True,timestamp=int(datetime(2026,10,5,tzinfo=timezone.utc).timestamp()))
    report = dict(output=str(output.resolve()),pandoc=str(pypandoc.get_pandoc_version()),
                  math_expressions_preserved=len(before),figures=figures,tables=tables,
                  warnings=[w.diagnostic for w in warnings])
    report_name = 'build-supplement.json' if args.supplement else 'build.json'
    (ROOT/'output'/report_name).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if warnings:
        raise RuntimeError('Resolve compiler warnings before delivery')


if __name__=='__main__':
    main()
