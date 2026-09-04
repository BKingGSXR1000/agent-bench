from pathlib import Path

root = Path(__file__).parents[1]
for name in ("index.html", "app.js", "styles.css"):
    assert (root / name).is_file(), name
html = (root / "index.html").read_text()
assert 'id="entry-form"' in html and 'id="entries"' in html
script = (root / "app.js").read_text()
assert "localStorage" in script and "render()" in script
print("baseline static checks passed")
