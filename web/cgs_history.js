/* ============================================================
 * CGS-导演台 (CGS-Director) · ComfyUI MiniMax H3 导演工作台
 * Author: zero14256
 * Repo:   https://github.com/zero14256/comfyui-MinimaxH3-CGS-Director
 * License: MIT · 请保留此来源标识
 * ============================================================
 */


import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/**
 * CGS-Director 生成历史（独立新增，不改动 cgs_timeline.js）
 * - 每次运行自动记录：时间 / 镜头数据(提示词、种子、参数) / 输出缩略图
 * - 右下角悬浮按钮打开面板：回看、手动删除单条、清空、自选存储路径
 */
const LS_PATH = "cgs_history_path";
const DEF_PATH = "D:/CGS_History";
const STYLE_ID = "cgs-history-style";

let panel = null;
let listEl = null;
let pathInput = null;
let busy = false;

function getPath() {
  try {
    return localStorage.getItem(LS_PATH) || DEF_PATH;
  } catch (e) {
    return DEF_PATH;
  }
}
function setPath(p) {
  try {
    localStorage.setItem(LS_PATH, p);
  } catch (e) {}
}

function installStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const st = document.createElement("style");
  st.id = STYLE_ID;
  st.textContent = `
.cgs-history-fab{position:fixed;right:18px;bottom:72px;z-index:29000;width:52px;height:52px;border-radius:14px;
  background:rgba(30,28,56,.92);border:1px solid rgba(123,47,247,.5);color:#c9b8ff;font-size:13px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;
  box-shadow:0 6px 22px rgba(0,0,0,.45);transition:all .15s;user-select:none;font-family:inherit;}
.cgs-history-fab:hover{background:rgba(60,44,110,.95);border-color:rgba(160,120,255,.8);}
.cgs-history-fab .ic{font-size:17px;line-height:1;}
.cgs-history-fab .tx{font-size:9px;line-height:1;}
.cgs-history-panel{position:fixed;right:18px;bottom:132px;z-index:29500;width:360px;max-height:64vh;
  background:#1a1830;border:1px solid rgba(123,47,247,.4);border-radius:12px;display:none;flex-direction:column;
  box-shadow:0 10px 36px rgba(0,0,0,.6);overflow:hidden;color:#e8e6f5;font-size:12px;font-family:inherit;}
.cgs-history-panel.open{display:flex;}
.cgs-history-head{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;
  background:rgba(123,47,247,.16);border-bottom:1px solid rgba(123,47,247,.25);font-weight:600;font-size:13px;}
.cgs-history-head .x{cursor:pointer;opacity:.7;padding:0 4px;}
.cgs-history-head .x:hover{opacity:1;}
.cgs-history-pathrow{display:flex;gap:6px;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.07);align-items:center;}
.cgs-history-pathrow input{flex:1;min-width:0;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);
  border-radius:6px;color:#fff;padding:4px 8px;font-size:11px;outline:none;}
.cgs-history-pathrow input:focus{border-color:rgba(123,47,247,.7);}
.cgs-history-pathrow button{background:rgba(123,47,247,.35);border:1px solid rgba(160,120,255,.4);color:#fff;
  border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;white-space:nowrap;}
.cgs-history-pathrow button:hover{background:rgba(123,47,247,.55);}
.cgs-history-tools{display:flex;gap:6px;padding:6px 12px;border-bottom:1px solid rgba(255,255,255,.07);}
.cgs-history-tools button{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);color:#cfcbe8;
  border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;}
.cgs-history-tools button:hover{background:rgba(255,255,255,.14);}
.cgs-history-tools button.danger:hover{background:rgba(220,60,60,.3);border-color:rgba(255,90,90,.5);}
.cgs-history-list{overflow-y:auto;padding:8px 12px;flex:1;display:flex;flex-direction:column;gap:8px;}
.cgs-history-item{display:flex;gap:8px;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.08);
  border-radius:8px;padding:7px;align-items:flex-start;}
.cgs-history-item .th{width:64px;height:64px;flex:none;border-radius:6px;background:#0d0c1a;overflow:hidden;
  display:flex;align-items:center;justify-content:center;font-size:10px;color:#666;}
.cgs-history-item .th img{width:100%;height:100%;object-fit:cover;}
.cgs-history-item .meta{flex:1;min-width:0;}
.cgs-history-item .t1{font-size:11px;color:#fff;font-weight:600;display:flex;align-items:center;gap:6px;}
.cgs-history-item .t2{font-size:10px;color:#9a95c0;margin-top:3px;line-height:1.5;word-break:break-all;}
.cgs-history-item .ops{margin-top:5px;display:flex;gap:6px;align-items:center;}
.cgs-history-item .ops button{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);color:#c9c4e6;
  border-radius:5px;padding:1px 8px;font-size:10px;cursor:pointer;}
.cgs-history-item .ops button.del:hover{background:rgba(220,60,60,.35);border-color:rgba(255,90,90,.55);color:#fff;}
.cgs-history-empty{padding:24px 8px;text-align:center;color:#7b76a3;font-size:11px;}
.cgs-history-note{padding:4px 12px 8px;color:#6f6a94;font-size:10px;}
`;
  document.head.appendChild(st);
}

/* ---------- 快照抓取 ---------- */
let pending = null;
let pendingImages = [];

function findDirectorNode() {
  const nodes = app.graph?.nodes || [];
  return nodes.find((n) => n.type === "CGSDirectorCore");
}

function makeEntry(d) {
  const now = new Date();
  const pad = (x) => String(x).padStart(2, "0");
  const id = now.getFullYear() + pad(now.getMonth() + 1) + pad(now.getDate()) + "_" +
    pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds());
  const time = now.getFullYear() + "-" + pad(now.getMonth() + 1) + "-" + pad(now.getDate()) + " " +
    pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds());
  const shots = Array.isArray(d.shots) ? d.shots : [];
  const nameOf = (x) => (typeof x === "string" ? x : x && x.name ? x.name : "");
  const mats = d.materials || {};
  return {
    id: id,
    time: time,
    mode: d.viewMode || (shots[0] && shots[0].mode) || "",
    shotCount: shots.length,
    shots: shots,
    global: d.global || null,
    seamless: d.seamless || null,
    materials: {
      images: (mats.images || []).map(nameOf),
      videos: (mats.videos || []).map(nameOf),
      audios: (mats.audios || []).map(nameOf),
    },
  };
}

function capture() {
  const cgs = findDirectorNode();
  if (!cgs || !cgs.widgets) return null;
  const w = cgs.widgets.find((x) => x.name === "shots_json");
  if (!w) return null;
  try {
    const d = JSON.parse(w.value);
    return makeEntry(d);
  } catch (e) {
    return null;
  }
}

async function saveEntry(entry, images) {
  try {
    const r = await api.fetchApi("/cgs_history/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: getPath(), entry: entry, images: images }),
    });
    const d = await r.json();
    console.log("[CGS History] save 完成:", JSON.stringify(d));
    refresh();
  } catch (e) {
    console.error("[CGS History] save failed", e);
  }
}

/* ---------- 面板 ---------- */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function fmtEntry(item) {
  const lines = [];
  lines.push((item.mode || "分镜") + " · " + (item.shotCount || 0) + " 镜头");
  const s0 = item.shots && item.shots[0];
  if (s0) lines.push("种子 " + (s0.seed !== undefined ? s0.seed : "-") + " · " + (s0.duration !== undefined ? s0.duration + "s" : ""));
  if (s0 && s0.prompt) {
    const p = s0.prompt.length > 46 ? s0.prompt.slice(0, 46) + "…" : s0.prompt;
    lines.push(p);
  }
  return lines.join("\n");
}

function renderItem(item) {
  const row = el("div", "cgs-history-item");
  const th = el("div", "th");
  const path = encodeURIComponent(getPath());
  if (item.images && item.images.length) {
    const img = document.createElement("img");
    img.src = "/cgs_history/thumb?path=" + path + "&name=" + encodeURIComponent(item.images[0]);
    img.loading = "lazy";
    th.appendChild(img);
  } else {
    th.textContent = "无图";
  }
  const meta = el("div", "meta");
  const t1 = el("div", "t1", item.time || item._file || "");
  const t2 = el("div", "t2", fmtEntry(item));
  const ops = el("div", "ops");
  const del = el("button", "del", "删除");
  del.onclick = async (e) => {
    e.stopPropagation();
    if (!confirm("删除这条生成记录？")) return;
    try {
      await api.fetchApi("/cgs_history/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: getPath(), name: item.id || item._file.replace(/\.json$/, "") }),
      });
      refresh();
    } catch (err) {
      console.error("[CGS History] delete failed", err);
    }
  };
  ops.appendChild(del);
  meta.appendChild(t1);
  meta.appendChild(t2);
  meta.appendChild(ops);
  row.appendChild(th);
  row.appendChild(meta);
  return row;
}

async function refresh() {
  if (!listEl) return;
  listEl.innerHTML = "";
  const empty = el("div", "cgs-history-empty", "暂无历史记录");
  try {
    const res = await api.fetchApi("/cgs_history/list?path=" + encodeURIComponent(getPath()));
    const data = await res.json();
    const items = (data && data.items) || [];
    if (!items.length) {
      listEl.appendChild(empty);
      return;
    }
    items.forEach((it) => listEl.appendChild(renderItem(it)));
  } catch (e) {
    empty.textContent = "读取失败：" + (e && e.message ? e.message : e);
    listEl.appendChild(empty);
  }
}

function openPanel() {
  panel.classList.add("open");
  if (pathInput) pathInput.value = getPath();
  refresh();
}
function closePanel() {
  panel.classList.remove("open");
}

function buildUI() {
  installStyle();
  const fab = el("div", "cgs-history-fab");
  fab.appendChild(el("span", "ic", "◉"));
  fab.appendChild(el("span", "tx", "历史"));
  fab.title = "导演台生成历史";
  fab.onclick = () => {
    if (panel.classList.contains("open")) closePanel();
    else openPanel();
  };

  panel = el("div", "cgs-history-panel");
  const head = el("div", "cgs-history-head");
  head.appendChild(el("span", "", "导演台生成历史"));
  const x = el("span", "x", "✕");
  x.onclick = closePanel;
  head.appendChild(x);

  const pathRow = el("div", "cgs-history-pathrow");
  pathInput = el("input");
  pathInput.type = "text";
  pathInput.value = getPath();
  pathInput.placeholder = "存储路径（如 D:/CGS_History）";
  pathInput.title = "存储路径，保存后新记录写入该目录";
  const savePath = el("button", "", "保存路径");
  savePath.onclick = () => {
    const v = (pathInput.value || "").trim();
    if (!v) return;
    setPath(v);
    refresh();
  };
  pathRow.appendChild(pathInput);
  pathRow.appendChild(savePath);

  const tools = el("div", "cgs-history-tools");
  const rf = el("button", "", "刷新");
  rf.onclick = refresh;
  const clr = el("button", "danger", "清空");
  clr.onclick = async () => {
    if (!confirm("确定清空整个历史目录？此操作不可恢复。")) return;
    try {
      await api.fetchApi("/cgs_history/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: getPath() }),
      });
      refresh();
    } catch (e) {
      console.error("[CGS History] clear failed", e);
    }
  };
  tools.appendChild(rf);
  tools.appendChild(clr);

  listEl = el("div", "cgs-history-list");
  const note = el("div", "cgs-history-note", "每次运行自动记录 · 缩略图 + 参数快照 · 可手动删除");
  panel.appendChild(head);
  panel.appendChild(pathRow);
  panel.appendChild(tools);
  panel.appendChild(listEl);
  panel.appendChild(note);

  document.body.appendChild(fab);
  document.body.appendChild(panel);

  // 外部点击关闭
  document.addEventListener("click", (e) => {
    if (panel.classList.contains("open") && !panel.contains(e.target) && !fab.contains(e.target)) {
      closePanel();
    }
  });
}

/* ---------- 注册 ---------- */
app.registerExtension({
  name: "CGS.DirectorHistory",
  async setup() {
    if (busy) return;
    busy = true;
    buildUI();

    api.addEventListener("execution_start", () => {
      pendingImages = [];
      pending = capture();
      console.log("[CGS History] execution_start capture:", pending ? "ok(" + pending.shots.length + "镜)" : "null(未找到导演台快照)");
    });
    api.addEventListener("executed", ({ detail }) => {
      if (!pending) return;
      const out = detail && detail.output;
      // 兼容 images / gifs(视频预览) / videos 等一切含 filename 的数组输出
      if (out && typeof out === "object") {
        Object.keys(out).forEach((k) => {
          const v = out[k];
          if (Array.isArray(v)) {
            v.forEach((item) => {
              if (item && typeof item === "object" && typeof item.filename === "string") {
                pendingImages.push(item);
              }
            });
          }
        });
      }
    });
    api.addEventListener("execution_success", () => {
      if (!pending) {
        console.log("[CGS History] execution_success 无快照，跳过");
        return;
      }
      const entry = pending;
      pending = null;
      if (!pendingImages.length) {
        console.log("[CGS History] execution_success 无输出图片/视频，跳过保存");
        return;
      }
      const imgs = pendingImages.slice();
      pendingImages = [];
      console.log("[CGS History] 保存记录，输出资源 " + imgs.length + " 个");
      saveEntry(entry, imgs);
    });
    api.addEventListener("execution_error", () => {
      pending = null;
      pendingImages = [];
    });
  },
});
