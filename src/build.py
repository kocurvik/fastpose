# -*- coding: utf-8 -*-
"""Build index.html from the page template and the benchmark data.

    python src/build.py

Reads src/page.html (the template, with a __DATA__ placeholder for the
results) and src/bench.json, and writes index.html at the repository root --
one self-contained file with the markup, styles, script and data inlined.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DESC = ('fastpose reaches PoseLib accuracy in a fraction of the runtime: a numba-compiled '
        'LO-RANSAC engine for robust relative pose, benchmarked on ETH3D, ScanNet++ and '
        'PhotoTourism.')

FAVICON = ('data:image/svg+xml,'
           '%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E'
           '%3Ctext y=%22.9em%22 font-size=%2290%22%3E%E2%9A%A1%3C/text%3E%3C/svg%3E')

HEAD = u"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta name="twitter:card" content="summary">
<link rel="icon" href="{favicon}">
{links}
<style>
  body{{margin:0}}
  img{{max-width:100%}}
  [hidden]{{display:none!important}}
</style>
</head>
<body>
"""


def build():
    page = io.open(os.path.join(HERE, 'page.html'), encoding='utf-8').read()
    data = io.open(os.path.join(HERE, 'bench.json'), encoding='utf-8').read()

    if '__DATA__' not in page:
        raise SystemExit('page.html has no __DATA__ placeholder to fill')
    body = page.replace('__DATA__', data)

    # the template carries <title> and the font <link>s at the top; both belong
    # in the document head this script builds around it
    title = re.search(r'<title>(.*?)</title>', body).group(1)
    body = body.replace('<title>%s</title>\n' % title, '', 1)

    links = re.findall(r'^<link [^>]*>\n', body, re.M)
    for link in links:
        body = body.replace(link, '', 1)

    head = HEAD.format(title=title, desc=DESC, favicon=FAVICON,
                       links=''.join(links).rstrip())
    out = head + body.lstrip() + u'\n</body>\n</html>\n'

    path = os.path.join(ROOT, 'index.html')
    io.open(path, 'w', encoding='utf-8').write(out)
    print('wrote %s (%.1f KB)' % (path, os.path.getsize(path) / 1024.0))


if __name__ == '__main__':
    build()
