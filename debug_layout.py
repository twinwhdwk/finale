"""Debug script for _force_system_layout and verovio breaks."""
import re
import verovio
from xml_to_systems import _read_mxl, _force_system_layout, _init_verovio

mxl = r'C:\Users\강우현\Desktop\Finale_Ref\xmls\I Have a Dream D (중등 음악1 천재).mxl'
measures_per_system = [6, 6, 5, 4, 5, 6]

xml_data = _read_mxl(mxl)
print("=== Original XML header ===")
print(repr(xml_data[:200]))
print()

modified = _force_system_layout(xml_data, measures_per_system)
print("=== Modified XML header ===")
print(repr(modified[:200]))
print()

# Check new-system elements
ns_matches = re.findall(r'<print[^>]*new-system[^>]*/>', modified)
print(f"new-system elements found: {len(ns_matches)}")
for m in ns_matches:
    print("  ", m)
print()

# Show measure 7 area
idx = modified.find('number="7"')
if idx >= 0:
    print("=== Measure 7 context ===")
    print(repr(modified[idx-10:idx+300]))
print()

# Test with breaks=encoded + systemMaxPerPage=1
_init_verovio()
tk = verovio.toolkit()
tk.setOptions({
    "pageWidth": 2800,
    "pageHeight": 600,
    "scale": 35,
    "adjustPageHeight": True,
    "systemMaxPerPage": 1,
    "breaks": "encoded",
    "footer": "none",
    "header": "none",
})
ok = tk.loadData(modified)
pages_with_max = tk.getPageCount()
print(f"breaks=encoded + systemMaxPerPage=1: ok={ok}, pages={pages_with_max}")

# Test with breaks=encoded only (no systemMaxPerPage)
tk2 = verovio.toolkit()
tk2.setOptions({
    "pageWidth": 2800,
    "pageHeight": 60000,
    "scale": 35,
    "adjustPageHeight": True,
    "breaks": "encoded",
    "footer": "none",
    "header": "none",
})
ok2 = tk2.loadData(modified)
pages_encoded_only = tk2.getPageCount()
print(f"breaks=encoded only (big page): ok={ok2}, pages={pages_encoded_only}")

# Test: save SVG and check for system groups
if pages_encoded_only >= 1:
    svg = tk2.renderToSVG(1)
    sys_count = svg.count('class="system"')
    print(f"  SVG system groups in page 1: {sys_count}")
    # Save for inspection
    with open(r'C:\Users\강우현\Desktop\Finale\debug_full.svg', 'w', encoding='utf-8') as f:
        f.write(svg)
    print("  Saved debug_full.svg")

# Test with breaks=smart
tk3 = verovio.toolkit()
tk3.setOptions({
    "pageWidth": 2800,
    "pageHeight": 600,
    "scale": 35,
    "adjustPageHeight": True,
    "systemMaxPerPage": 1,
    "breaks": "smart",
    "footer": "none",
    "header": "none",
})
ok3 = tk3.loadData(xml_data)  # original, no modification
pages_smart = tk3.getPageCount()
print(f"breaks=smart + systemMaxPerPage=1 (original XML): ok={ok3}, pages={pages_smart}")

# Test with auto breaks
tk4 = verovio.toolkit()
tk4.setOptions({
    "pageWidth": 2800,
    "pageHeight": 600,
    "scale": 35,
    "adjustPageHeight": True,
    "systemMaxPerPage": 1,
    "breaks": "auto",
    "footer": "none",
    "header": "none",
})
ok4 = tk4.loadData(xml_data)
pages_auto = tk4.getPageCount()
print(f"breaks=auto + systemMaxPerPage=1 (original XML): ok={ok4}, pages={pages_auto}")
