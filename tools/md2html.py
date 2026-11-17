# -*- coding: utf-8 -*-
"""把 技术报告.md 转成符合 html-to-doc 版式规范的单个 HTML。

规范要点（来自 doc-formatter §2.5）：
  正文 SimSun 12pt + text-indent:2em
  章节标题必须用 h1~h3（否则目录收录不到）
  封面标题禁用 h1（会被 TOC 收成第一条）
  页码走 CSS @page 的 margin-box
  目录用 nav.doc-toc，独占一个 section role=cover
  正文第一个 section 带 data-page-restart="1"
  图片落 images/，相对路径
"""

import os
import re
import html as H

BASE = r"D:/Programming/Machine_Learning/couplet"
SRC = os.path.join(BASE, "docs", "技术报告.md")
OUT = os.path.join(BASE, "output", "20261117-couplet", "stage2",
                   "formatted-对联生成模型.html")

lines = open(SRC, encoding="utf-8").read().split("\n")

# ---------------- 解析 ----------------
blocks = []          # (type, payload)
i = 0
title = ""
while i < len(lines):
    raw = lines[i]
    s = raw.rstrip()

    # 代码块
    if s.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i])
            i += 1
        i += 1
        blocks.append(("code", "\n".join(buf)))
        continue

    # 空行
    if not s.strip():
        i += 1
        continue

    # 分隔线
    if s.strip() == "---":
        i += 1
        continue

    # 标题
    m = re.match(r"^(#{1,4})\s+(.*)$", s)
    if m:
        level = len(m.group(1))
        text = m.group(2).strip()
        if level == 1:
            title = text
        else:
            blocks.append(("h%d" % (level - 1), text))
        i += 1
        continue

    # 图片
    m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s.strip())
    if m:
        alt, path = m.group(1), m.group(2)
        name = os.path.basename(path)
        blocks.append(("img", (alt, name)))
        i += 1
        continue

    # 表格
    if s.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
        header = [c.strip() for c in s.strip().strip("|").split("|")]
        i += 2
        rows = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        blocks.append(("table", (header, rows)))
        continue

    # 引用
    if s.lstrip().startswith(">"):
        buf = []
        while i < len(lines) and lines[i].lstrip().startswith(">"):
            buf.append(lines[i].lstrip().lstrip(">").strip())
            i += 1
        blocks.append(("quote", " ".join(buf)))
        continue

    # 有序 / 无序列表
    if re.match(r"^\s*([-*]|\d+\.)\s+", s):
        items = []
        ordered = bool(re.match(r"^\s*\d+\.", s))
        while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
            items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]).strip())
            i += 1
        blocks.append(("ol" if ordered else "ul", items))
        continue

    # 普通段落（合并连续行）
    buf = [s.strip()]
    i += 1
    while i < len(lines):
        nxt = lines[i]
        if (not nxt.strip() or nxt.startswith("```") or re.match(r"^#{1,4}\s", nxt)
                or nxt.lstrip().startswith("|") or nxt.lstrip().startswith(">")
                or re.match(r"^\s*([-*]|\d+\.)\s+", nxt)
                or re.match(r"^!\[", nxt.strip()) or nxt.strip() == "---"):
            break
        buf.append(nxt.strip())
        i += 1
    blocks.append(("p", " ".join(buf)))


def inline(t):
    """行内标记：**粗体**、`代码`

    尖括号换成全角：转换引擎会把 &lt; 和 &#60; 都解码成 < 再当标签解析，
    导致 <pad> 这类技术文本丢内容（实测 C-1）。全角不会被解析。
    """
    t = t.replace("&", "&amp;")
    t = t.replace("<", "＜").replace(">", "＞")
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


# ---------------- 目录 ----------------
toc = []
_h1_no = 0
_h2_no = 0
for kind, payload in blocks:
    if kind == "h1":
        _h1_no += 1
        _h2_no = 0
        toc.append((1, payload, "section-%d" % _h1_no))
    elif kind == "h2":
        _h2_no += 1
        toc.append((2, payload, "section-%d-%d" % (_h1_no, _h2_no)))

# ---------------- 输出 ----------------
parts = []
parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>%s</title>
<style>
/* 页面规则：正文页脚居中页码，封面与目录页无页码 */
@page { @bottom-center { content: counter(page); } }
@page cover { @bottom-center { content: none; } }
body { font-family: "SimSun", "宋体", serif; font-size: 12pt; line-height: 1.75; color: #1a1a1a; }
section[role="cover"] { page: cover; }
p { text-indent: 2em; margin: 0.5em 0; }
h1 { font-family: "SimSun", "宋体", serif; font-size: 17pt; font-weight: bold;
     color: #8B2E2E; margin: 1.6em 0 0.7em; text-indent: 0; border-bottom: 1px solid #d8cfc0; padding-bottom: 0.28em; }
h2 { font-family: "SimSun", "宋体", serif; font-size: 14pt; font-weight: bold;
     color: #2b2b2b; margin: 1.1em 0 0.5em; text-indent: 0; }
h3 { font-family: "SimSun", "宋体", serif; font-size: 12.5pt; font-weight: bold;
     color: #2b2b2b; margin: 0.9em 0 0.4em; text-indent: 0; }
table { border-collapse: collapse; width: 100%%; margin: 0.9em 0; font-size: 10.5pt; }
th { border: 1px solid #999; background: #f3efe6; padding: 5px 7px; font-weight: bold; text-align: left; }
td { border: 1px solid #999; padding: 5px 7px; vertical-align: top; }
ul, ol { margin: 0.5em 0 0.5em 1.6em; }
li { margin: 0.22em 0; }
blockquote { margin: 0.8em 0; padding: 0.55em 1em; background: #faf7f0;
             border-left: 3px solid #c8b79a; text-indent: 0; }
blockquote p { text-indent: 0; }
pre { font-family: "Consolas", "Courier New", monospace; font-size: 9.5pt; line-height: 1.5;
      background: #f7f5f0; border: 1px solid #e0d8c8; padding: 0.7em 0.9em; margin: 0.8em 0;
      white-space: pre-wrap; }
code { font-family: "Consolas", "Courier New", monospace; font-size: 10pt;
       background: #f2efe8; padding: 0 3px; }
pre code { background: none; padding: 0; font-size: 9.5pt; }
figure { margin: 1em 0; text-align: center; text-indent: 0; }
figure img { max-width: 100%%; }
figcaption { font-size: 10pt; color: #666; margin-top: 0.4em; text-indent: 0; }
.cover-title { font-size: 26pt; font-weight: bold; text-align: center;
               margin: 5.5em 0 0.6em; text-indent: 0; letter-spacing: 0.04em; }
.cover-sub { font-size: 12pt; text-align: center; color: #555; text-indent: 0; margin: 0.3em 0; }
.cover-rule { border: none; border-top: 2px solid #8B2E2E; width: 30%%; margin: 1.5em auto; }
.toc-title { font-size: 16pt; font-weight: bold; text-align: center; text-indent: 0; margin: 0 0 1.2em; }
.toc-list, .toc-list ol { list-style: none; padding-left: 0; margin: 0; }
.toc-list li { text-indent: 0; margin: 0.42em 0; font-size: 11pt; }
.toc-list ol li { padding-left: 1.8em; font-size: 10.5pt; color: #444; }
.toc-list a { color: #1a1a1a; text-decoration: none; }
</style>
</head>
<body>
""" % H.escape(title))

# 封面
parts.append('<section role="cover">')
parts.append('<p class="cover-title">%s</p>' % H.escape(title.replace("：", "：")))
parts.append('<hr class="cover-rule">')
parts.append('<p class="cover-sub">seq2seq Transformer · 810,496 参数</p>')
parts.append('<p class="cover-sub">2026-11-17</p>')
parts.append('</section>')

# 目录（独占一节）
parts.append('<section role="cover">')
parts.append('<nav class="doc-toc" aria-label="文档目录">')
parts.append('<p class="toc-title">目　录</p>')
parts.append('<ol class="toc-list">')
open_ol = False
for idx, (level, text, sid) in enumerate(toc):
    if level == 1:
        if open_ol:
            parts.append("</ol></li>")
            open_ol = False
        parts.append('<li><a href="#%s">%s</a>' % (sid, H.escape(text)))
        if idx + 1 < len(toc) and toc[idx + 1][0] == 2:
            parts.append("<ol>")
            open_ol = True
        else:
            parts.append("</li>")
    else:
        parts.append('<li><a href="#%s">%s</a></li>' % (sid, H.escape(text)))
if open_ol:
    parts.append("</ol></li>")
parts.append("</ol>")
parts.append("</nav>")
parts.append("</section>")

# 正文（另起一节，强制从新页开始）
parts.append('<section role="body" data-page-restart="1" style="page-break-before: always;">')
h1_count = 0
h2_count = 0
for kind, payload in blocks:
    if kind == "h1":
        h1_count += 1
        h2_count = 0
        parts.append('<h1 id="section-%d">%s</h1>' % (h1_count, inline(payload)))
    elif kind == "h2":
        h2_count += 1
        parts.append('<h2 id="section-%d-%d">%s</h2>'
                     % (h1_count, h2_count, inline(payload)))
    elif kind == "h3":
        parts.append("<h3>%s</h3>" % inline(payload))
    elif kind == "p":
        parts.append("<p>%s</p>" % inline(payload))
    elif kind == "quote":
        parts.append("<blockquote><p>%s</p></blockquote>" % inline(payload))
    elif kind == "code":
        parts.append("<pre><code>%s</code></pre>" % H.escape(payload))
    elif kind in ("ul", "ol"):
        tag = kind
        parts.append("<%s>" % tag)
        for it in payload:
            parts.append("<li>%s</li>" % inline(it))
        parts.append("</%s>" % tag)
    elif kind == "table":
        header, rows = payload
        parts.append("<table>")
        parts.append("<tr>%s</tr>" % "".join("<th>%s</th>" % inline(c) for c in header))
        for r in rows:
            parts.append("<tr>%s</tr>" % "".join("<td>%s</td>" % inline(c) for c in r))
        parts.append("</table>")
    elif kind == "img":
        alt, name = payload
        # 用 base64 内嵌：转换引擎不解析相对路径的 <img src>
        ipath = os.path.join(BASE, "output", "20261117-couplet",
                             "stage2", "images", name)
        try:
            import base64
            with open(ipath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            parts.append('<figure><img src="data:image/png;base64,%s" alt="%s">'
                         '<figcaption>%s</figcaption></figure>'
                         % (b64, H.escape(alt), H.escape(alt)))
        except OSError:
            parts.append('<p>[图缺失：%s]</p>' % H.escape(name))
parts.append("</section>")
parts.append("</body></html>")

html = "\n".join(parts)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print("已输出 %s" % OUT)
print("大小 %.1f KB" % (len(html.encode("utf-8")) / 1024))
print("块统计：")
from collections import Counter
print("  ", dict(Counter(k for k, _ in blocks)))
print("目录条目 %d 个（一级 %d / 二级 %d）"
      % (len(toc), sum(1 for l, _, _ in toc if l == 1), sum(1 for l, _, _ in toc if l == 2)))
