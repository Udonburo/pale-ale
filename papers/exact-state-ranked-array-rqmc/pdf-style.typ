#set document(title: manuscript-title, author: "Aoi Kawasaki", keywords: ("Array-RQMC", "exact simulation", "digital nets", "state aggregation", "implicit execution"))
#set page(paper: "a4", margin: (left: 22mm, right: 22mm, top: 21mm, bottom: 21mm), footer: context align(center)[#text(size: 8.5pt, fill: rgb("#666666"))[#counter(page).display("1")]])
#set page(header: context {
  if counter(page).get().first() > 1 {
    align(right, text(size: 8pt, fill: rgb("#666666"), manuscript-running-title))
  }
})
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: .55em, spacing: .7em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 13pt, weight: "bold", hyphenate: false)
#show heading.where(level: 1): set block(above: 1.2em, below: .55em)
#show heading.where(level: 2): set text(size: 11.3pt, weight: "bold", hyphenate: false)
#show heading: it => { set block(sticky: true); it }
#show link: set text(fill: rgb("#234C70"))
#set math.equation(numbering: manuscript-equation-numbering)
#show math.equation.where(block: true): set block(above: .65em, below: .65em)
#set table(inset: (x: 4pt, y: 4pt), stroke: none, fill: (x,y) => if y == 0 { rgb("#F1F3F5") } else { none })
#show table: set text(size: 9pt, hyphenate: false)
#show table: set par(justify: false)
#show table.cell.where(y: 0): set text(weight: "bold")
#show table.hline: set table.hline(stroke: .5pt + rgb("#9099A1"))
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
#show raw.where(block: true): it => block(width: 100%, breakable: false, fill: rgb("#F4F5F6"), inset: 9pt, {
  set par(leading: 6.3pt, spacing: 0pt)
  it
})
#let keep(body) = block(breakable: false, above: .55em, below: .55em, body)
#align(center)[
  #set par(justify: false)
  #text(size: 21pt, weight: "bold", hyphenate: false, manuscript-title)
  #v(.65em)
  #text(size: 11pt)[Aoi Kawasaki]
  #v(.3em)
  #text(size: 9pt, fill: rgb("#555555"))[5 October 2026]
]
#v(.8em)
