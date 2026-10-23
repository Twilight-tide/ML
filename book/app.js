/* 全书共用脚本：目录注入、当前章高亮、已读标记、阅读进度 */

const BOOK = [
  { file: "index.html",    no: "",  title: "封面与阅读指南", group: "卷首" },
  { file: "ch01.html",     no: "1", title: "先跑起来",           group: "第一部 · 地基" },
  { file: "ch02.html",     no: "2", title: "把数学翻译成代码",   group: "第一部 · 地基" },
  { file: "ch03.html",     no: "3", title: "机器学习的最小闭环", group: "第一部 · 地基" },
  { file: "ch04.html",     no: "4", title: "神经网络与反向传播", group: "第一部 · 地基" },
  { file: "ch05.html",     no: "5", title: "语言模型",           group: "第二部 · 语言" },
  { file: "ch06.html",     no: "6", title: "注意力的由来",       group: "第二部 · 语言" },
  { file: "ch07.html",     no: "7", title: "多头注意力",         group: "第三部 · Transformer" },
  { file: "ch08.html",     no: "8", title: "位置、残差与前馈",   group: "第三部 · Transformer" },
  { file: "ch09.html",     no: "9", title: "拼装与训练",         group: "第三部 · Transformer" },
  { file: "ch10.html",     no: "10", title: "精读论文",          group: "第四部 · 验证" },
  { file: "appendix.html", no: "A", title: "附录：速查表",       group: "第四部 · 验证" },
  { file: "numpy.html",    no: "B", title: "numpy 够用指南",     group: "第四部 · 验证" }
];

const READ_KEY = "ml-book-read-v1";
let readSet = new Set();
try { readSet = new Set(JSON.parse(localStorage.getItem(READ_KEY) || "[]")); } catch (e) {}

const here = (location.pathname.split("/").pop() || "index.html").toLowerCase();

/* ---------- 注入侧栏目录 ---------- */
function buildToc() {
  const nav = document.getElementById("toc");
  if (!nav) return;
  const list = document.createElement("div");
  list.className = "toc-list";

  BOOK.forEach(item => {
    if (item.group) {
      const g = document.createElement("div");
      g.className = "toc-group";
      g.textContent = item.group;
      list.appendChild(g);
    }
    const a = document.createElement("a");
    a.href = item.file;
    if (item.file.toLowerCase() === here) a.classList.add("active");
    if (readSet.has(item.file)) a.classList.add("done");
    a.innerHTML = `<span class="num">${item.no}</span><span class="ttl">${item.title}</span>`;
    list.appendChild(a);
  });
  nav.appendChild(list);
}

/* ---------- 上下章导航 ---------- */
function buildChapNav() {
  const nav = document.getElementById("chapnav");
  if (!nav) return;
  const i = BOOK.findIndex(b => b.file.toLowerCase() === here);
  if (i < 0) return;
  const prev = BOOK[i - 1], next = BOOK[i + 1];
  let html = "";
  if (prev) {
    html += `<a href="${prev.file}"><span class="dir">← 上一章</span><span class="ttl">${prev.no ? prev.no + ". " : ""}${prev.title}</span></a>`;
  } else {
    html += `<span style="flex:1"></span>`;
  }
  if (next) {
    html += `<a class="next" href="${next.file}"><span class="dir">下一章 →</span><span class="ttl">${next.no ? next.no + ". " : ""}${next.title}</span></a>`;
  }
  nav.innerHTML = html;
}

/* ---------- 已读标记 ---------- */
function buildMarkRead() {
  const btn = document.getElementById("markread");
  if (!btn) return;
  const sync = () => {
    const on = readSet.has(here);
    btn.classList.toggle("on", on);
    btn.textContent = on ? "✓ 本章已读" : "○ 标记本章已读";
  };
  btn.addEventListener("click", () => {
    if (readSet.has(here)) readSet.delete(here); else readSet.add(here);
    try { localStorage.setItem(READ_KEY, JSON.stringify([...readSet])); } catch (e) {}
    sync();
  });
  sync();
}

/* ---------- 阅读进度条 ---------- */
function trackProgress() {
  const bar = document.getElementById("readbar");
  if (!bar) return;
  const upd = () => {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    const p = h > 0 ? (window.scrollY / h) * 100 : 0;
    bar.style.width = Math.min(100, Math.max(0, p)) + "%";
  };
  window.addEventListener("scroll", upd, { passive: true });
  window.addEventListener("resize", upd);
  upd();
}

buildToc();
buildChapNav();
buildMarkRead();
trackProgress();
