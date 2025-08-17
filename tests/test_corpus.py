from loyalty_spend.llm.corpus import chunk_markdown, load_corpus

DOC = """# Title

Intro paragraph.

## Part A

One two three four five.

Six seven eight nine ten.

```
# not a heading
```

## Part B

Short.
"""


def test_chunks_follow_headings_and_respect_the_size():
    chunks = chunk_markdown(DOC, "doc.md", max_words=6)
    headings = [c.heading for c in chunks]
    assert headings[0] == "Title"
    assert headings.count("Part A") == 3
    assert headings[-1] == "Part B"
    assert all(len(c.text.split()) <= 6 or "\n\n" not in c.text for c in chunks)
    assert any("# not a heading" in c.text for c in chunks)
    assert chunks[1].citation == "doc.md, Part A"
    assert len({c.id for c in chunks}) == len(chunks)


def test_project_documentation_is_loaded():
    chunks = load_corpus()
    sources = {c.source for c in chunks}
    assert {"README.md", "docs/methodology.md"} <= sources
    assert len(chunks) > 50
