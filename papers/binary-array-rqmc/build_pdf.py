"""Typeset main.md without changing its scientific content or running experiments."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
READER = "markdown+tex_math_dollars-implicit_figures"
WIDTHS = [(30,70), (13,21,22,16,28), (10,19,29,22,20),
          (15,23,13,15,20,14), (40,30,30), (16,28,25,31),
          (9,15,12,15,21,14,14)]


def plain(inlines):
    out=[]
    for n in inlines:
        if n["t"]=="Str":
            out.append(n["c"])
        elif n["t"] in ("Space","SoftBreak","LineBreak"):
            out.append(" ")
        elif n["t"] in ("Strong","Emph","SmallCaps","Strikeout"):
            out.append(plain(n["c"]))
    return "".join(out)


def raw(text):
    return {"t":"RawBlock","c":["typst",text]}


def typography(node):
    # The Markdown source is unchanged. Normalize typographic dashes in prose
    # for PDF portability; mathematical minus signs and TeX are untouched.
    if isinstance(node,list):
        return [typography(x) for x in node]
    if not isinstance(node,dict):
        return node
    node=deepcopy(node)
    if node.get("t")=="Str":
        node["c"]=node["c"].translate(str.maketrans({"\u2011":"-","\u2013":"-","\u2014":"-"}))
    elif "c" in node:
        node["c"]=typography(node["c"])
    return node


def math_nodes(node):
    if isinstance(node,list):
        return sum((math_nodes(x) for x in node),[])
    if isinstance(node,dict):
        if node.get("t")=="Math":
            return [node["c"]]
        return math_nodes(node.get("c",[]))
    return []


def code_blocks(node):
    if isinstance(node,list):
        return sum((code_blocks(x) for x in node),[])
    if isinstance(node,dict):
        if node.get("t")=="CodeBlock":
            return [node["c"][1]]
        return code_blocks(node.get("c",[]))
    return []


def grouped_blocks(blocks):
    out=[]
    i=0
    table_count=0
    image_count=0
    while i<len(blocks):
        b=blocks[i]
        if b["t"]=="Header":
            b["c"][0]-=1
            # Keep the short appendix together on a fresh page instead of
            # separating its encoding definition from the introductory line.
            if plain(b["c"][2]).startswith("Appendix "):
                out.append(raw("#pagebreak(weak: true)"))
            if plain(b["c"][2])=="References":
                out.extend([b,raw("#set text(size: 9pt)\n#set par(leading: .45em, spacing: .4em)")])
                i+=1
                continue
        title=plain(b["c"]) if b["t"]=="Para" else ""
        if title.startswith("Table ") and i+1<len(blocks) and blocks[i+1]["t"]=="Table":
            out.extend([raw("#keep["),b,blocks[i+1],raw("]")])
            i+=2
            continue
        if b["t"]=="Para" and any(x["t"]=="Image" for x in b["c"]):
            assert i+1<len(blocks) and blocks[i+1]["t"]=="Para"
            assert plain(blocks[i+1]["c"]).startswith("Figure ")
            # One unbreakable image+caption group, with no automatic duplicate caption.
            out.extend([raw("#keep["),b,raw("#set text(size: 9pt)\n#set par(justify: false)"),
                        blocks[i+1],raw("]")])
            image_count+=1
            i+=2
            continue
        if b["t"]=="Para" and i+1<len(blocks) and blocks[i+1]["t"]=="CodeBlock":
            out.extend([raw("#keep["),b,blocks[i+1],raw("]")])
            i+=2
            continue
        out.append(b)
        i+=1
    assert image_count==2
    # Tables may now be nested only between raw delimiters, not inside AST Divs.
    for b in out:
        if b["t"]=="Table":
            specs=b["c"][2]
            widths=WIDTHS[table_count]
            assert len(specs)==len(widths)
            for spec,width in zip(specs,widths):
                if spec[0]["t"]=="AlignDefault":
                    spec[0]={"t":"AlignLeft"}
                spec[1]={"t":"ColWidth","c":width/100}
            table_count+=1
    assert table_count==len(WIDTHS)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir",type=Path,help="Optional directory from pip install --target.")
    parser.add_argument("--output",type=Path,default=ROOT/"output/pdf/binary-array-rqmc.pdf")
    args=parser.parse_args()
    if args.tools_dir:
        sys.path.insert(0,str(args.tools_dir.resolve()))
    import pypandoc
    import typst

    markdown=(ROOT/"main.md").read_text(encoding="utf-8")
    ast=json.loads(pypandoc.convert_text(markdown,"json",format=READER))
    original_math=math_nodes(ast["blocks"])
    original_code=code_blocks(ast["blocks"])
    assert ast["blocks"][0]["t"]=="Header" and ast["blocks"][1]["t"]=="Para"
    title=plain(ast["blocks"][0]["c"][2])
    byline=ast["blocks"][1]["c"]
    linebreak=next(i for i,x in enumerate(byline) if x["t"]=="LineBreak")
    author=plain(byline[:linebreak])
    date=plain(byline[linebreak+1:]).replace("\u2014","-")
    # Reproducible metadata follows the manuscript date, not the build clock.
    date_text=date.split(" - ",1)[1]
    timestamp=int(datetime.strptime(date_text,"%d %B %Y").replace(tzinfo=timezone.utc).timestamp())
    ast["blocks"]=grouped_blocks(typography(ast["blocks"][2:]))
    assert original_math==math_nodes(ast["blocks"])
    assert original_code==code_blocks(ast["blocks"])
    # The AST transformation changes layout, never the mathematical expressions.
    body=pypandoc.convert_text(json.dumps(ast),"typst",format="json",
                              extra_args=["--wrap=none"]).replace("\r\n","\n")
    # Inline images have no intrinsic width in the writer; constrain them to the page.
    body=body.replace('image("figures/component-costs.png")',
                      'image("/figures/component-costs.png", width: 100%)')
    body=body.replace('image("figures/reuse-costs.png")',
                      'image("/figures/reuse-costs.png", width: 100%)')
    if 'image("figures/' in body:
        raise ValueError("Unexpected image syntax: review the writer conversion.")
    # Resolve the companion link relative to the PDF, not the Markdown source.
    companion=os.path.relpath(ROOT/"repro/README.md",args.output.resolve().parent)
    companion=Path(companion).as_posix()
    body=body.replace('#link("repro/README.md")',
                      '#link('+json.dumps(companion)+')')
    # Pandoc 3.9's Typst writer emits a full negative em for TeX's negative
    # thin space. Restore -1/6 em; this is typography, not mathematical content.
    body=body.replace('#h(-1em)', '#h(-0.166667em)')
    # In this manuscript TeX's literal bitwise ampersand is not an alignment
    # marker. Scope the fix to math so raw pseudocode retains a literal &.
    body=body.replace('"parity" \\( a & r \\)', '"parity" \\( a \\& r \\)')
    body=body.replace('$&$', '$\\&$')
    # TeX double bars are norm delimiters, not Typst's parallel relation.
    body=body.replace('parallel hat(Delta) - Delta parallel_2^2',
                      'norm(hat(Delta) - Delta)_2^2')
    body=body.replace('parallel z_i - z_j parallel_2^2',
                      'norm(z_i - z_j)_2^2')
    if 'parallel' in body:
        raise ValueError("Unconverted norm: review the writer conversion.")
    for code in original_code:
        if code not in body:
            raise ValueError("Pseudocode changed during typesetting.")
    prefix="\n".join("#let "+name+" = "+json.dumps(value,ensure_ascii=False)
                     for name,value in (("manuscript-title",title),("manuscript-author",author),("manuscript-date",date)))
    source=prefix+"\n"+(ROOT/"pdf-style.typ").read_text(encoding="utf-8")+"\n"+body
    generated=ROOT/"tmp/pdfs/manuscript.typ"
    generated.parent.mkdir(parents=True,exist_ok=True)
    generated.write_text(source,encoding="utf-8")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    _,warnings=typst.compile_with_warnings(generated,output=args.output,root=ROOT,
                                          ignore_system_fonts=True,
                                          timestamp=timestamp)
    if warnings:
        raise RuntimeError("\n".join(w.diagnostic for w in warnings))
    print(json.dumps({"output":str(args.output.resolve()),
                      "pandoc":str(pypandoc.get_pandoc_version()),
                      "math_expressions_preserved":len(original_math),
                      "pseudocode_blocks_preserved":len(original_code),
                      "tables":len(WIDTHS),"figures":2,"compiler_warnings":0},indent=2))


if __name__=="__main__":
    main()
