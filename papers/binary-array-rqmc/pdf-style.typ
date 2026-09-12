// Layout only. Scientific text, equations, and numerical values come from main.md.
#set document(
  title: manuscript-title,
  author: manuscript-author,
  keywords: ("Array-RQMC", "Markov chains", "binary innovations", "order maintenance"),
)
#set page(
  paper: "a4",
  margin: (left: 22mm, right: 22mm, top: 21mm, bottom: 21mm),
  numbering: "1",
  number-align: center,
  footer: context align(center)[
    #text(size: 8.5pt, fill: rgb("#666666"))[#counter(page).display("1")]
  ],
)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: .55em, spacing: .7em, first-line-indent: 0em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 13pt, weight: "bold", hyphenate: false)
#show heading.where(level: 1): set block(above: 1.25em, below: .6em)
#show heading.where(level: 2): set text(size: 11.3pt, weight: "bold", hyphenate: false)
#show heading.where(level: 2): set block(above: 1em, below: .5em)
#show heading: it => {
  set block(sticky: true)
  it
}
#show link: set text(fill: rgb("#234C70"))
#set math.equation(numbering: none)
#show math.equation.where(block: true): set block(above: .7em, below: .7em)
#set table(
  inset: (x: 4pt, y: 4pt),
  stroke: none,
  fill: (x,y) => if y == 0 { rgb("#F1F3F5") } else { none },
)
#show table: set text(size: 9pt, hyphenate: false)
#show table: set par(justify: false)
#show table.cell.where(y: 0): set text(weight: "bold")
#show table.hline: set table.hline(stroke: .5pt + rgb("#9099A1"))
#show figure.where(kind: table): set block(above: .4em, below: .7em)
#show raw: set text(font: "DejaVu Sans Mono", size: 7.2pt)
#show raw.where(block: true): it => block(
  width: 100%, breakable: false,
  fill: rgb("#F4F5F6"), inset: 9pt, above: .6em, below: .6em,
  it,
)
#let keep(body) = block(breakable: false, above: .65em, below: .65em, body)
#align(center)[
  #set par(justify: false)
  #text(size: 21pt, weight: "bold", hyphenate: false, manuscript-title)
  #v(.6em)
  #text(size: 11pt, manuscript-author)
  #v(.3em)
  #text(size: 9pt, fill: rgb("#555555"), manuscript-date)
]
#v(.8em)
