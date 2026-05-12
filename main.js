// Main frontend logic (formerly inline in frontend.html).
//
// Loaded as an external file to avoid environments that silently block/skip
// very large inline scripts.
//
// IMPORTANT: `frontend.html` may attempt both module + classic loading.
// Wrap everything in an IIFE with a one-time lock to avoid double init.
;(function () {
  if (window.__MAIN_SCRIPT_LOADED__) return;
  window.__MAIN_SCRIPT_LOADED__ = true;
  window.__MAIN_JS_BUILD__ = "2026-04-21.3";
  // Make sure ES5 fallback sees "started" as early as possible.
  window.__MAIN_SCRIPT_STARTED__ = true;

    (function () {
      window.__REACHED_BOTTOM__ = true;
      var el = document.getElementById("jsCompatBanner");
      if (el && !window.__MAIN_SCRIPT_STARTED__) {
        el.textContent = "已解析到页面底部，准备启动主脚本…";
      }
    })();
    // 主脚本启动“探针”：即便后续报错，也要能诊断到错误原因
    window.__MAIN_SCRIPT_BOOT__ = true;
    window.__MAIN_SCRIPT_OK__ = false;
    window.__MAIN_SCRIPT_ERROR__ = "";
    try {
    const API_BASE = "http://127.0.0.1:8000";

    // --- Conversation memory (frontend-side session plumbing) ---
    function getOrCreateUserId() {
      try {
        const k = "philo_user_id";
        let v = (localStorage.getItem(k) || "").trim();
        if (!v) {
          v = "u_" + Math.random().toString(16).slice(2) + Date.now().toString(16);
          localStorage.setItem(k, v);
        }
        return v;
      } catch (e) {
        return "default";
      }
    }

    function getConversationId() {
      try {
        return (localStorage.getItem("philo_conversation_id") || "").trim();
      } catch (e) {
        return "";
      }
    }

    function setConversationId(v) {
      try {
        if (!v) return;
        localStorage.setItem("philo_conversation_id", String(v));
      } catch (e) {}
    }

    // --- Multi-session store (local-only) ---
    const SESSIONS_KEY = "philo_conversations_v1";
    function _loadSessions() {
      try {
        const raw = (localStorage.getItem(SESSIONS_KEY) || "").trim();
        if (!raw) return [];
        const arr = JSON.parse(raw);
        return Array.isArray(arr) ? arr : [];
      } catch (e) {
        return [];
      }
    }
    function _saveSessions(arr) {
      try {
        localStorage.setItem(SESSIONS_KEY, JSON.stringify(Array.isArray(arr) ? arr : []));
      } catch (e) {}
    }
    function _nowIso() {
      try { return new Date().toISOString(); } catch (e) { return String(Date.now()); }
    }
    function _newConvId() {
      return Math.random().toString(16).slice(2) + Date.now().toString(16);
    }
    function ensureSessionExists(convId) {
      const id = (convId || "").trim();
      if (!id) return;
      const list = _loadSessions();
      const found = list.find((x) => x && String(x.id || "") === id);
      if (found) return;
      list.unshift({ id, title: "新会话", updatedAt: _nowIso() });
      _saveSessions(list.slice(0, 80));
    }
    function updateSessionUpdatedAt(convId) {
      const id = (convId || "").trim();
      if (!id) return;
      const list = _loadSessions();
      for (let i = 0; i < list.length; i++) {
        const it = list[i];
        if (it && String(it.id || "") === id) {
          it.updatedAt = _nowIso();
          list.splice(i, 1);
          list.unshift(it);
          _saveSessions(list.slice(0, 80));
          return;
        }
      }
      list.unshift({ id, title: "新会话", updatedAt: _nowIso() });
      _saveSessions(list.slice(0, 80));
    }
    function getSessionTitle(convId) {
      const id = (convId || "").trim();
      if (!id) return "";
      const list = _loadSessions();
      const it = list.find((x) => x && String(x.id || "") === id);
      return it ? String(it.title || "") : "";
    }
    function setSessionTitle(convId, title) {
      const id = (convId || "").trim();
      const t = String(title || "").trim();
      if (!id || !t) return;
      const list = _loadSessions();
      for (let i = 0; i < list.length; i++) {
        const it = list[i];
        if (it && String(it.id || "") === id) {
          it.title = t;
          it.updatedAt = _nowIso();
          list.splice(i, 1);
          list.unshift(it);
          _saveSessions(list.slice(0, 80));
          renderSessionSelect();
          return;
        }
      }
      list.unshift({ id, title: t, updatedAt: _nowIso() });
      _saveSessions(list.slice(0, 80));
      renderSessionSelect();
    }

    async function autoNameCurrentSession(opts) {
      const o = opts || {};
      const convId = getConversationId();
      if (!convId) return;
      const curTitle = getSessionTitle(convId);
      const onlyIfDefault = (o.onlyIfDefault !== false);
      if (onlyIfDefault) {
        const t0 = (curTitle || "").trim();
        if (t0 && t0 !== "新会话" && t0 !== "未命名会话") return;
      }
      const q = String(o.question || "").trim();
      const a = String(o.answer || "").trim();
      if (!q && !a) return;
      try {
        if (memoryStatus) memoryStatus.textContent = "会话标题生成中…";
        const resp = await fetch(`${API_BASE}/api/session/suggest_title`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, answer: a }),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        const title = data && data.title ? String(data.title).trim() : "";
        if (title) setSessionTitle(convId, title);
      } catch (e) {
        // ignore
      }
    }
    function renderSessionSelect() {
      const el = document.getElementById("sessionSelect");
      if (!el) return;
      const current = getConversationId();
      if (current) ensureSessionExists(current);
      const list = _loadSessions();
      el.innerHTML = "";
      if (!list.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "（暂无会话，点“新建”）";
        opt.disabled = true;
        el.appendChild(opt);
        return;
      }
      list.forEach((s) => {
        const id = String((s && s.id) || "");
        if (!id) return;
        const opt = document.createElement("option");
        opt.value = id;
        const title = String((s && s.title) || "").trim() || "未命名会话";
        opt.textContent = title.length > 36 ? (title.slice(0, 36) + "…") : title;
        if (id === current) opt.selected = true;
        el.appendChild(opt);
      });
    }
    function switchSession(id) {
      const cid = String(id || "").trim();
      if (!cid) return;
      setConversationId(cid);
      ensureSessionExists(cid);
      renderSessionSelect();
      try { refreshMemoryViewer(); } catch (e) {}
    }

    async function seedSessionsFromDisk() {
      try {
        async function _list(uid) {
          const r = await fetch(`${API_BASE}/api/conversations/list?user_id=${encodeURIComponent(uid)}`);
          if (!r.ok) return [];
          const j = await r.json();
          const arr = (j && j.conversation_ids && Array.isArray(j.conversation_ids)) ? j.conversation_ids : [];
          return arr.map((x) => String(x || "").trim()).filter((x) => x);
        }

        let ids = await _list(USER_ID);
        // If current USER_ID has no history but default does, switch back to default.
        if (!ids.length && USER_ID !== "default") {
          const idsDefault = await _list("default");
          if (idsDefault.length) {
            try {
              localStorage.setItem("philo_user_id", "default");
              location.reload();
              return;
            } catch (e0) {}
          }
        }
        if (!ids.length) return;
        const list = _loadSessions();
        const seen = new Set(list.map((x) => (x && x.id) ? String(x.id) : ""));
        let changed = false;
        ids.forEach((id) => {
          const cid = String(id || "").trim();
          if (!cid || seen.has(cid)) return;
          list.push({ id: cid, title: "未命名会话", updatedAt: _nowIso() });
          changed = true;
        });
        if (changed) {
          // keep most recent at front roughly by backend ordering
          const merged = [];
          ids.forEach((id) => {
            const cid = String(id || "").trim();
            const it = list.find((x) => x && String(x.id || "") === cid);
            if (it) merged.push(it);
          });
          list.forEach((it) => {
            if (!it) return;
            const cid = String(it.id || "");
            if (!cid) return;
            if (!merged.find((x) => String(x.id || "") === cid)) merged.push(it);
          });
          _saveSessions(merged.slice(0, 200));
        }
      } catch (e) {
        // ignore
      }
    }

    // --- Collapsible sidebars (push, icon-only) ---
    function _getBool(key, defVal) {
      try {
        const v = (localStorage.getItem(key) || "").trim();
        if (v === "") return defVal;
        return v === "1" || v.toLowerCase() === "true";
      } catch (e) {
        return defVal;
      }
    }
    function _setBool(key, v) {
      try { localStorage.setItem(key, v ? "1" : "0"); } catch (e) {}
    }
    function applySidebarState() {
      const lc = _getBool("philo_ui_left_collapsed", false);
      const rc = _getBool("philo_ui_right_collapsed", false);
      const leftbar = document.getElementById("leftbar");
      const rightbar = document.querySelector(".sidebar");
      const tl = document.getElementById("toggleLeftbarBtn");
      const tr = document.getElementById("toggleRightbarBtn");
      if (leftbar) leftbar.classList.toggle("collapsed", !!lc);
      if (rightbar) rightbar.classList.toggle("collapsed", !!rc);
      if (tl) tl.textContent = lc ? "⟩" : "⟨";
      if (tr) tr.textContent = rc ? "⟨" : "⟩";
    }
    function toggleLeftbar() {
      const next = !_getBool("philo_ui_left_collapsed", false);
      _setBool("philo_ui_left_collapsed", next);
      applySidebarState();
    }
    function toggleRightbar() {
      const next = !_getBool("philo_ui_right_collapsed", false);
      _setBool("philo_ui_right_collapsed", next);
      applySidebarState();
    }

    const USER_ID = getOrCreateUserId();
    const _compatEl = document.getElementById("jsCompatBanner");
    if (_compatEl) _compatEl.textContent = "主脚本已启动（build=" + (window.__MAIN_JS_BUILD__ || "unknown") + "）";
    window.__MAIN_SCRIPT_STARTED__ = true;

    const ingestButton = document.getElementById("ingestButton");
    const profileSelect = document.getElementById("profileSelect");
    const ingestStatus = document.getElementById("ingestStatus");
    const refreshLibrariesBtn = document.getElementById("refreshLibrariesBtn");
    const toggleLibraryPanelBtn = document.getElementById("toggleLibraryPanelBtn");
    const libraryPanel = document.getElementById("libraryPanel");
    const libraryList = document.getElementById("libraryList");
    const loadSourcesBtn = document.getElementById("loadSourcesBtn");
    const librarySourcesList = document.getElementById("librarySourcesList");
    const refreshTrashBtn = document.getElementById("refreshTrashBtn");
    const trashList = document.getElementById("trashList");
    const detectLibraryLangBtn = document.getElementById("detectLibraryLangBtn");
    const libraryLangStatus = document.getElementById("libraryLangStatus");
    const libraryLangProfile = document.getElementById("libraryLangProfile");
    const sessionSelect = document.getElementById("sessionSelect");
    const newSessionBtn = document.getElementById("newSessionBtn");
    const autoNameSessionBtn = document.getElementById("autoNameSessionBtn");
    const renameSessionBtn = document.getElementById("renameSessionBtn");
    const deleteSessionBtn = document.getElementById("deleteSessionBtn");
    const refreshMemoryBtn = document.getElementById("refreshMemoryBtn");
    const compactWikiBtn = document.getElementById("compactWikiBtn");
    const memoryStatus = document.getElementById("memoryStatus");
    const memoryPaths = document.getElementById("memoryPaths");
    const memoryHistory = document.getElementById("memoryHistory");
    const memoryHistoryPanel = document.getElementById("memoryHistoryPanel");
    const toggleMemoryHistoryBtn = document.getElementById("toggleMemoryHistoryBtn");
    const memoryWiki = document.getElementById("memoryWiki");
    const uploadDropzone = document.getElementById("uploadDropzone");
    const uploadFileInput = document.getElementById("uploadFileInput");
    const libraryIdInput = document.getElementById("libraryIdInput");
    const autoIngestAfterUpload = document.getElementById("autoIngestAfterUpload");
    const ingestProgressWrap = document.getElementById("ingestProgressWrap");
    const ingestProgress = document.getElementById("ingestProgress");
    const ingestProgressLabel = document.getElementById("ingestProgressLabel");
    const questionInput = document.getElementById("questionInput");
    const askButton = document.getElementById("askButton");
    const answerContent = document.getElementById("answerContent");
    const answerModelBadge = document.getElementById("answerModelBadge");
    const statusText = document.getElementById("statusText");
    const excerptList = document.getElementById("excerptList");
    const keywordHitList = document.getElementById("keywordHitList");
    const keywordHitsSubtitle = document.getElementById("keywordHitsSubtitle");
    const keywordSourceStats = document.getElementById("keywordSourceStats");
    const sidebarSubtitle = document.getElementById("sidebarSubtitle");
    const excerptDetailCard = document.getElementById("excerptDetailCard");
    const excerptDetailTitle = document.getElementById("excerptDetailTitle");
    const excerptDetailText = document.getElementById("excerptDetailText");
    const debugBar = document.getElementById("debugBar");
    const debugPanel = document.getElementById("debugPanel");
    const autoB2PreviewSelect = document.getElementById("autoB2PreviewSelect");
    const autoB2PreviewStatus = document.getElementById("autoB2PreviewStatus");
    const autoB2PreviewOutput = document.getElementById("autoB2PreviewOutput");
    const keywordInput = document.getElementById("keywordInput");
    const sourceFilterInput = document.getElementById("sourceFilterInput");
    const autoKeywords = document.getElementById("autoKeywords");
    const useHybrid = document.getElementById("useHybrid");
    const useRerank = document.getElementById("useRerank");
    const useSepReference = document.getElementById("useSepReference");
    const ultraLongAnswer = document.getElementById("ultraLongAnswer");
    const answerStyleSelect = document.getElementById("answerStyleSelect");
    const answerModelSelect = document.getElementById("answerModelSelect");

    let currentDocs = [];
    let currentKeywordHitDocs = [];
    let librariesCache = [];

    // --- Persist library selection (checkbox + weight) ---
    const LIB_PICK_STORE_KEY = "philo_selected_libraries_v1";

    function _loadLibPicks() {
      try {
        const raw = localStorage.getItem(LIB_PICK_STORE_KEY) || "";
        if (!raw) return {};
        const obj = JSON.parse(raw);
        return (obj && typeof obj === "object") ? obj : {};
      } catch (e) {
        return {};
      }
    }

    function _saveLibPicks(obj) {
      try {
        localStorage.setItem(LIB_PICK_STORE_KEY, JSON.stringify(obj || {}));
      } catch (e) {}
    }

    function _libFullKey(profile, libraryId) {
      const p = (profile || "").trim() || "quality";
      const lid = (libraryId || "").trim() || "default";
      return `${p}__${lid}`;
    }

    function persistCurrentLibSelections() {
      try {
        const store = _loadLibPicks();
        // store per-profile selections to avoid collisions
        const next = (store && typeof store === "object") ? store : {};
        document.querySelectorAll("input[data-libkey]").forEach((el) => {
          const lid = el.getAttribute("data-libkey") || "";
          const prof = el.getAttribute("data-libprofile") || "";
          const full = _libFullKey(prof, lid);
          const weightEl = document.getElementById(`w_${lid}`);
          const w = weightEl ? Number(weightEl.value || 1) : 1;
          if (el.checked) {
            next[full] = { checked: true, weight: (Number.isFinite(w) && w > 0 ? w : 1) };
          } else {
            // keep if previously existed? choose to remove when unchecked
            if (next[full]) delete next[full];
          }
        });
        _saveLibPicks(next);
      } catch (e) {}
    }

    function restoreLibSelectionsIntoDom() {
      const store = _loadLibPicks();
      document.querySelectorAll("input[data-libkey]").forEach((el) => {
        const lid = el.getAttribute("data-libkey") || "";
        const prof = el.getAttribute("data-libprofile") || "";
        const full = _libFullKey(prof, lid);
        const st = store ? store[full] : null;
        if (st && st.checked) {
          el.checked = true;
          const weightEl = document.getElementById(`w_${lid}`);
          if (weightEl && st.weight != null) {
            const ww = Number(st.weight);
            if (Number.isFinite(ww) && ww > 0) weightEl.value = String(ww);
          }
        }
      });
    }

    function libSourcesSel() {
      return document.getElementById("librarySourcesSelect");
    }

    function currentLibraryKey() {
      const sel = libSourcesSel();
      if (!sel) return "";
      const v = (sel.value || "").trim();
      return v && v.indexOf("__") >= 0 ? v : "";
    }

    function _fmtJson(obj) {
      try {
        return JSON.stringify(obj, null, 2);
      } catch (e) {
        return String(obj);
      }
    }

    /** 下拉框仅一个禁用项：加载中 / 失败说明（避免一直显示「等待加载」却无任何提示） */
    function setLibraryDropdownMessage(text) {
      const sel = libSourcesSel();
      if (!sel) return;
      sel.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.disabled = true;
      opt.textContent = text;
      sel.appendChild(opt);
    }

    function getSelectedLibraries() {
      const picks = [];
      document.querySelectorAll("input[data-libkey]").forEach((el) => {
        if (el.checked) {
          const key = el.getAttribute("data-libkey");
          const prof = el.getAttribute("data-libprofile") || "";
          const lid = el.getAttribute("data-libid") || key || "";
          const weightEl = document.getElementById(`w_${key}`);
          const w = weightEl ? Number(weightEl.value || 1) : 1;
          picks.push({
            key,
            weight: Number.isFinite(w) && w > 0 ? w : 1,
            profile: prof,
            library_id: lid,
          });
        }
      });
      return picks;
    }

    async function loadLibraries() {
      setStatus("库", "正在连接后端并刷新向量库列表…");
      setLibraryDropdownMessage("正在连接 " + API_BASE + " …");
      const ac = new AbortController();
      const t = window.setTimeout(function () {
        ac.abort();
      }, 20000);
      try {
        const resp = await fetch(`${API_BASE}/api/libraries`, { signal: ac.signal });
        window.clearTimeout(t);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();
        librariesCache = data.libraries || [];
        renderLibraries(librariesCache);
        renderLibrarySourcesSelect(librariesCache);
        // NOTE: do NOT auto-overwrite `libraryLangProfile` here.
        // That panel is used to show "selected libraries" summary and must stay stable.
        if (librariesCache.length > 0) {
          try {
            await loadLibrarySources();
          } catch (e2) {
            console.error(e2);
            setStatus("库", "库列表已加载，但文献列表请求失败：" + (e2 && e2.message ? e2.message : String(e2)));
          }
        } else if (librarySourcesList) {
          librarySourcesList.innerHTML = "";
          const empty = document.createElement("div");
          empty.style.fontSize = "12px";
          empty.style.color = "#6b7280";
          empty.textContent = "未发现任何向量库目录（data/chroma_db_*）。请先 Ingest。";
          librarySourcesList.appendChild(empty);
        }
        setStatus("库", `已加载 ${librariesCache.length} 个库`);
      } catch (e) {
        window.clearTimeout(t);
        console.error(e);
        if (libraryList) libraryList.textContent = "加载库列表失败。请确认已用本目录 start_app 或 uvicorn 启动后端。";
        const aborted = e && (e.name === "AbortError" || (e.message && String(e.message).indexOf("aborted") >= 0));
        const tip = aborted
          ? "（连接超时 20s：后端未响应，请先启动 start_app）"
          : "（无法连接后端：" + (e && e.message ? e.message : String(e)) + "。请启动后再点「刷新库列表」）";
        setLibraryDropdownMessage(tip);
        setStatus("库失败", aborted ? "连接后端超时或未完成，请确认 127.0.0.1:8000 已监听" : e && e.message ? e.message : String(e));
      }
    }

    async function loadTrashItems() {
      if (!trashList) return;
      trashList.textContent = "正在加载回收站…";
      try {
        const resp = await fetch(`${API_BASE}/api/trash/items`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const items = data.items || [];
        trashList.innerHTML = "";
        if (!items.length) {
          trashList.textContent = "回收站为空（或尚无带 manifest 的删除记录）。";
          return;
        }
        items.forEach((it) => {
          const row = document.createElement("div");
          row.className = "excerpt-item";
          row.style.cursor = "default";
          const meta = document.createElement("div");
          meta.className = "excerpt-meta";
          meta.textContent = `${it.trash_session} · ${it.profile}__${it.library_id} · ${it.source || ""}`;
          const detail = document.createElement("div");
          detail.className = "excerpt-text";
          detail.style.marginTop = "4px";
          detail.style.color = "#4b5563";
          detail.textContent = it.original_path || "";
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "btn secondary";
          btn.style.marginTop = "6px";
          btn.textContent = "恢复文献并重建索引";
          btn.addEventListener("click", async () => {
            const ok = confirm(
              `将文件移回原始路径并写入当前向量库索引？\n\n${it.original_path || ""}\n\n若目标路径已有同名文件，将中止。`
            );
            if (!ok) return;
            try {
              const r = await fetch(`${API_BASE}/api/trash/${encodeURIComponent(it.trash_session)}/restore`, {
                method: "POST",
              });
              const txt = await r.text();
              if (!r.ok) {
                let msg = txt;
                try {
                  const j = JSON.parse(txt);
                  msg = j.detail || msg;
                } catch (e2) {}
                throw new Error(msg || `HTTP ${r.status}`);
              }
              await loadTrashItems();
              await loadLibraries();
              const libKey = `${it.profile}__${it.library_id}`;
              const lss = libSourcesSel();
              if (lss) {
                let found = false;
                for (let oi = 0; oi < lss.options.length; oi++) {
                  if (lss.options[oi].value === libKey) {
                    found = true;
                    break;
                  }
                }
                if (found) {
                  lss.value = libKey;
                  await loadLibrarySources();
                }
              }
              setStatus("恢复", "已完成恢复与索引");
            } catch (e) {
              console.error(e);
              alert(`恢复失败：${e.message || e}`);
            }
          });
          row.appendChild(meta);
          row.appendChild(detail);
          row.appendChild(btn);
          trashList.appendChild(row);
        });
      } catch (e) {
        console.error(e);
        trashList.textContent = "加载回收站失败。";
      }
    }

    function renderLibraries(libs) {
      if (!libraryList) return;
      libraryList.innerHTML = "";
      (libs || []).forEach((l) => {
        const key = `${l.profile}__${l.library_id}`;
        const row = document.createElement("div");
        row.className = "excerpt-item";
        row.style.cursor = "default";
        const line1 = document.createElement("div");
        line1.className = "excerpt-meta";
        const cc = (l.chroma_collection_used || l.chroma_collection || "");
        const ccSuffix = cc ? (" · chroma=" + cc) : "";
        line1.textContent = `${key}  (chunks=${(l.chunks_count == null ? 0 : l.chunks_count)}, sources=${(l.sources_count == null ? 0 : l.sources_count)})${ccSuffix}`;
        line1.style.cursor = "pointer";
        line1.title = "点击：在下方下拉框选中该库并加载文献列表";
        line1.addEventListener("click", () => {
          const sel = libSourcesSel();
          if (!sel) return;
          sel.value = key;
          loadLibrarySources();
          setStatus("库", "已加载文献列表：" + key);
        });
        const line2 = document.createElement("div");
        line2.className = "excerpt-text";
        line2.innerHTML = `
          <label style="display:flex; gap:8px; align-items:center; margin-top:6px;">
            <input type="checkbox" data-libkey="${l.library_id}" data-libprofile="${l.profile}" data-libid="${l.library_id}" />
            参与检索（library_id=${l.library_id}）
          </label>
          <label style="display:flex; gap:8px; align-items:center; margin-top:6px;">
            权重 <input id="w_${l.library_id}" type="number" value="1" min="0.1" step="0.1" style="width:80px; padding:4px 6px; border:1px solid #d1d5db; border-radius:6px;" />
          </label>
        `;
        row.appendChild(line1);
        row.appendChild(line2);
        libraryList.appendChild(row);
      });
      // Restore persisted selections after re-render
      restoreLibSelectionsIntoDom();
      // Bind change listeners once (event delegation) to persist changes
      try {
        if (!libraryList.__picksBound) {
          libraryList.__picksBound = true;
          libraryList.addEventListener("change", (e) => {
            const t = e && e.target ? e.target : null;
            if (!t) return;
            // checkbox or weight input changes
            if (t.matches && (t.matches("input[data-libkey]") || (t.id && String(t.id).startsWith("w_")))) {
              persistCurrentLibSelections();
            }
          });
        }
      } catch (e0) {}
    }

    function renderLibrarySourcesSelect(libs) {
      const sel = libSourcesSel();
      if (!sel) return;
      sel.innerHTML = "";
      const list = libs || [];
      if (!list.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.disabled = true;
        opt.textContent = "（未发现 chroma 库，请先 Ingest）";
        sel.appendChild(opt);
        return;
      }
      list.forEach((l) => {
        const opt = document.createElement("option");
        opt.value = `${l.profile}__${l.library_id}`;
        opt.textContent = `${l.profile}__${l.library_id}`;
        sel.appendChild(opt);
      });
      if (sel.options.length > 0) {
        sel.selectedIndex = 0;
      }
    }

    async function detectSelectedLibrariesLanguage() {
      const picks = getSelectedLibraries();
      if (!picks.length) {
        if (libraryLangStatus) libraryLangStatus.textContent = "请先在上方库列表勾选 1 个或多个库。";
        if (libraryLangProfile) libraryLangProfile.textContent = "language_profile：暂无（未勾选库）";
        return;
      }
      // Lock the view so loadLibraries() won't overwrite the selected-libraries summary.
      try { window.__LIB_LANG_PROFILE_LOCKED__ = true; } catch (e0) {}
      if (libraryLangStatus) libraryLangStatus.textContent = `正在检测…（count=${picks.length}）`;
      if (libraryLangProfile) libraryLangProfile.textContent = "language_profile：检测中…";

      const result = {};
      const needCompute = [];
      for (const p of picks) {
        const prof = p.profile || "quality";
        const lid = p.library_id || p.key || "default";
        const k = `${prof}__${lid}`;
        const hit = (librariesCache || []).find((x) => `${x.profile}__${x.library_id}` === k);
        if (hit && hit.language_profile) {
          result[k] = { cached: true, language_profile: hit.language_profile };
        } else {
          needCompute.push({ profile: prof, library_id: lid, key: k });
        }
      }

      for (const it of needCompute) {
        try {
          const resp = await fetch(
            `${API_BASE}/api/libraries/${encodeURIComponent(it.profile)}/${encodeURIComponent(it.library_id)}/language_profile`
          );
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          const data = await resp.json();
          result[it.key] = {
            cached: true,
            sampled_chunks: data.sampled_chunks,
            updated_at_ts: data.updated_at_ts,
            language_profile: data.language_profile || null,
          };
        } catch (e) {
          result[it.key] = { cached: false, error: (e && e.message) ? e.message : String(e) };
        }
      }

      if (libraryLangProfile) libraryLangProfile.textContent = "language_profile（selected）:\n" + _fmtJson(result);
      if (libraryLangStatus) libraryLangStatus.textContent = "已更新（显示已勾选库）";
      // Refresh libraries cache (language_profile may have been computed), but keep the locked view.
      try { await loadLibraries(); } catch (e2) {}
    }

    async function refreshMemoryViewer() {
      if (!memoryStatus) return;
      const conversation_id = getConversationId();
      memoryStatus.textContent = "正在加载…";
      try {
        const resp = await fetch(`${API_BASE}/api/memory/view?user_id=${encodeURIComponent(USER_ID)}&conversation_id=${encodeURIComponent(conversation_id || "")}`);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();
        if (memoryPaths) memoryPaths.textContent = "paths:\n" + _fmtJson(data.paths || {});
        if (memoryHistory) memoryHistory.textContent = "history:\n" + _fmtJson(data.history || []);
        // 全文展示，与磁盘 user.md 一致（不做 trim，避免首尾空白被误删）
        const wikiRaw = data.wiki_md != null ? String(data.wiki_md) : "";
        const wikiLen = typeof data.wiki_chars === "number" ? data.wiki_chars : wikiRaw.length;
        if (memoryWiki) memoryWiki.textContent = wikiRaw.length ? wikiRaw : "（wiki 为空，user.md 还没写入稳定信息）";
        memoryStatus.textContent =
          "已刷新 · user.md " + (wikiLen ? "全文 " + wikiLen + " 字符" : "空文件");
      } catch (e) {
        memoryStatus.textContent = "加载失败：" + (e && e.message ? e.message : String(e));
      }
    }

    async function compactWikiNow() {
      if (!memoryStatus) return;
      const ok = confirm(
        "确认立即整合 Wiki？\n\n- 将调用模型对当前 user.md 做整理\n- 规则：不新增任何内容，只删除冗余/重复并调整顺序\n- 完成后会将写入计数归零\n"
      );
      if (!ok) return;
      memoryStatus.textContent = "正在整合 Wiki…";
      let compactStatus = "";
      try {
        const resp = await fetch(`${API_BASE}/api/wiki/compact`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: USER_ID }),
        });
        const txt = await resp.text();
        if (!resp.ok) throw new Error(txt || ("HTTP " + resp.status));
        let data = null;
        try { data = JSON.parse(txt); } catch (e) { data = { ok: true }; }
        if (data && data.ok) {
          const r = (data && data.result) ? data.result : {};
          const ch = (r && typeof r.changed !== "undefined") ? (r.changed ? "有变化" : "无变化") : "";
          const b = (r && r.before_bullets != null) ? ` bullets ${r.before_bullets}→${r.after_bullets}` : "";
          compactStatus = "整合完成（已归零计数）" + (ch ? ` · ${ch}` : "") + b;
          memoryStatus.textContent = compactStatus;
        } else {
          const detail = (data && data.result && data.result.detail) ? String(data.result.detail) : "";
          compactStatus = "整合失败（已归零计数；未覆盖 Wiki）" + (detail ? ("：" + detail) : "");
          memoryStatus.textContent = compactStatus;
        }
      } catch (e) {
        compactStatus = "整合失败：" + (e && e.message ? e.message : String(e));
        memoryStatus.textContent = compactStatus;
      }
      try { await refreshMemoryViewer(); } catch (e2) {}
      if (compactStatus) memoryStatus.textContent = compactStatus + " · Wiki 已刷新";
    }

    async function loadLibrarySources() {
      const sel = libSourcesSel();
      if (!sel) return;
      if (sel.options.length && sel.selectedIndex < 0) {
        sel.selectedIndex = 0;
      }
      const v = (sel.value || "").trim();
      if (!v || v.indexOf("__") < 0) {
        if (librarySourcesList) {
          librarySourcesList.innerHTML = "";
          const hint = document.createElement("div");
          hint.style.fontSize = "12px";
          hint.style.color = "#6b7280";
          hint.textContent = "请先在上方点击「刷新库列表」，待下拉框出现 quality__… 等待选项后再试。";
          librarySourcesList.appendChild(hint);
        }
        setStatus("文献", "请先在列表中选中一个库（当前下拉值为空）。");
        return;
      }
      const parts = v.split("__");
      const profile = parts[0] || "quality";
      const library_id = parts.slice(1).join("__") || "default";
      setStatus("文献", "正在加载「" + profile + "__" + library_id + "」的文献列表…");
      if (librarySourcesList) {
        librarySourcesList.innerHTML = "";
        const loading = document.createElement("div");
        loading.className = "excerpt-text";
        loading.style.fontSize = "12px";
        loading.style.color = "#6b7280";
        loading.textContent = "正在请求文献列表…（若长时间无结果请确认后端已启动）";
        librarySourcesList.appendChild(loading);
        try {
          librarySourcesList.scrollIntoView({ block: "nearest", behavior: "smooth" });
        } catch (eScroll) {}
      }
      const ac = new AbortController();
      const t = window.setTimeout(function () {
        ac.abort();
      }, 25000);
      try {
        const resp = await fetch(
          `${API_BASE}/api/libraries/${encodeURIComponent(profile)}/${encodeURIComponent(library_id)}/sources`,
          { signal: ac.signal }
        );
        window.clearTimeout(t);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();
        renderLibrarySourcesList(profile, library_id, data.sources || []);
        setStatus("文献", "已加载文献条目（见侧栏「库内文献」下方）");
      } catch (e) {
        window.clearTimeout(t);
        console.error(e);
        const aborted = e && (e.name === "AbortError" || (e.message && String(e.message).indexOf("aborted") >= 0));
        if (librarySourcesList) {
          librarySourcesList.innerHTML = "";
          const err = document.createElement("div");
          err.style.fontSize = "12px";
          err.style.color = "#b45309";
          err.style.lineHeight = "1.5";
          err.textContent =
            "加载失败：" +
            (aborted
              ? "请求超时（25s），请确认后端在本机 127.0.0.1:8000 运行。"
              : e && e.message
                ? e.message
                : String(e));
          librarySourcesList.appendChild(err);
        }
        setStatus("文献失败", aborted ? "请求超时" : e && e.message ? e.message : String(e));
      }
    }

    function renderLibrarySourcesList(profile, library_id, sources) {
      if (!librarySourcesList) return;
      librarySourcesList.innerHTML = "";
      if (!sources || !sources.length) {
        const empty = document.createElement("div");
        empty.style.fontSize = "12px";
        empty.style.color = "#6b7280";
        empty.style.lineHeight = "1.5";
        empty.innerHTML =
          "当前库<strong>没有</strong>可在列表中展示的文献（稀疏索引 FTS 中无记录）。<br />" +
          "常见原因：① 尚未成功运行 Ingest；② 仅有 Chroma 目录但 <code>sparse_fts_*.db</code> 为空或不存在；③ 库名与数据不一致。请先对该 profile 重新执行一次 Ingest。";
        librarySourcesList.appendChild(empty);
        return;
      }
      (sources || []).forEach((s) => {
        const item = document.createElement("div");
        item.className = "excerpt-item";
        const meta = document.createElement("div");
        meta.className = "excerpt-meta";
        meta.textContent = `${s.source}  (chunks=${(s.chunks == null ? 0 : s.chunks)})`;
        const btn = document.createElement("button");
        btn.className = "btn secondary";
        btn.style.marginTop = "6px";
        btn.textContent = "删除该文献";
        btn.addEventListener("click", async () => {
          const ok = confirm(`确认删除？\n- library: ${profile}__${library_id}\n- source: ${s.source}\n\n此操作会从索引删除，并把原文件移动到 data/trash/。`);
          if (!ok) return;
          try {
            const prep = await fetch(`${API_BASE}/api/libraries/${encodeURIComponent(profile)}/${encodeURIComponent(library_id)}/sources/${encodeURIComponent(s.source)}/prepare_delete`, { method: "POST" });
            if (!prep.ok) throw new Error(`prepare HTTP ${prep.status}`);
            const prepData = await prep.json();
            const tok = prepData.confirm_token;
            const del = await fetch(`${API_BASE}/api/libraries/${encodeURIComponent(profile)}/${encodeURIComponent(library_id)}/sources/${encodeURIComponent(s.source)}?confirm_token=${encodeURIComponent(tok)}`, { method: "DELETE" });
            if (!del.ok) {
              const t = await del.text();
              throw new Error(t || `delete HTTP ${del.status}`);
            }
            await loadLibrarySources();
          } catch (e) {
            console.error(e);
            alert(`删除失败：${e.message || e}`);
          }
        });
        item.appendChild(meta);
        item.appendChild(btn);
        librarySourcesList.appendChild(item);
      });
    }

    // --- Minimal in-page diagnostics (so you don't need DevTools) ---
    function setStatus(kind, msg) {
      if (!statusText) return;
      const prefix = kind ? `[${kind}] ` : "";
      statusText.textContent = prefix + (msg || "");
    }
    window.addEventListener("error", (e) => {
      const msg = (e && e.message) ? e.message : String(e);
      setStatus("JS错误", msg);
    });
    window.addEventListener("unhandledrejection", (e) => {
      const reason = e && e.reason ? e.reason : e;
      const msg = (reason && reason.message) ? reason.message : String(reason);
      setStatus("Promise错误", msg);
    });

    function renderDebug(data) {
      if (!data) {
        debugPanel.textContent = "暂无调试信息。";
        if (autoB2PreviewOutput) autoB2PreviewOutput.textContent = "（回答后可在此查看）";
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      const lines = [];
      lines.push(`term_source: ${(data.term_source == null ? "question" : data.term_source)}`);
      lines.push(`profile: ${(data.profile == null ? "quality" : data.profile)}`);
      lines.push(`keywords_used: ${(data.keywords_used || []).join(", ") || "(none)"}`);
      lines.push(`source_filters_used: ${(data.source_filters_used || []).join(", ") || "(none)"}`);
      lines.push(`user_terms_used: ${(data.user_terms_used || []).join(", ") || "(none)"}`);
      lines.push(`auto_terms_used: ${(data.auto_terms_used || []).join(", ") || "(none)"}`);
      lines.push(`dropped_terms: ${(data.dropped_terms || []).join(", ") || "(none)"}`);
      lines.push(`keyword_query: ${data.keyword_query || "(empty)"}`);
      lines.push(`hybrid: ${data.hybrid ? "true" : "false"}`);
      lines.push(`reranked: ${data.reranked ? "true" : "false"}`);
      if (typeof data.sep_reference_enabled !== "undefined") {
        lines.push(`sep_reference_enabled: ${data.sep_reference_enabled ? "true" : "false"}`);
        lines.push(`sep_docs_kept: ${(data.sep_docs_kept == null ? 0 : data.sep_docs_kept)}`);
        lines.push(`sep_max_docs: ${(data.sep_max_docs == null ? 0 : data.sep_max_docs)}`);
        lines.push(`sep_weight: ${(data.sep_weight == null ? 0 : data.sep_weight)}`);
      }
      lines.push(`answer_style: ${data.answer_style || "哲学论述"}`);
      lines.push(`answer_model: ${data.answer_model || "(unknown)"}`);
      lines.push(`answer_max_output_tokens: ${(data.answer_max_output_tokens == null ? 0 : data.answer_max_output_tokens)}`);
      const dbg = data.debug || {};
      // Store latest debug payload for Auto+B2 preview panel
      try {
        window.__LAST_DEBUG__ = dbg;
      } catch (e0) {}
      lines.push(`dense_top_ids: ${(dbg.dense_top_ids || []).join(", ") || "(none)"}`);
      lines.push(`sparse_top_ids: ${(dbg.sparse_top_ids || []).join(", ") || "(none)"}`);
      lines.push(`fused_top_ids: ${(dbg.fused_top_ids || []).join(", ") || "(none)"}`);
      if (dbg.sep_dense_top_ids) {
        lines.push(`sep_dense_top_ids: ${(dbg.sep_dense_top_ids || []).join(", ") || "(none)"}`);
      }
      lines.push(`retrieved_before_rerank: ${(dbg.retrieved_before_rerank == null ? 0 : dbg.retrieved_before_rerank)}`);
      lines.push(`final_docs: ${(dbg.final_docs == null ? 0 : dbg.final_docs)}`);
      lines.push(`keyword_hits_count: ${(dbg.keyword_hits_count == null ? 0 : dbg.keyword_hits_count)}`);
      lines.push(`keyword_source_stats_top10: ${JSON.stringify(dbg.keyword_source_stats_top10 || [])}`);
      lines.push(`coverage: ${JSON.stringify(dbg.coverage || {})}`);
      lines.push(`final_k_effective: ${(dbg.final_k == null ? 0 : dbg.final_k)}`);
      lines.push(`final_k_cap: ${(dbg.final_k_cap == null ? 0 : dbg.final_k_cap)}`);
      lines.push(`chroma_collection: ${dbg.chroma_collection || "(unknown)"}`);
      lines.push(`chroma_path: ${dbg.chroma_path || "(unknown)"}`);
      lines.push(`sparse_db_path: ${dbg.sparse_db_path || "(unknown)"}`);
      debugPanel.textContent = lines.join("\n");
      // Refresh Auto+B2 preview output if user selected a view
      try {
        if (autoB2PreviewSelect) {
          updateAutoB2Preview(autoB2PreviewSelect.value, dbg);
        }
      } catch (e1) {}
    }

    function updateAutoB2Preview(mode, dbg) {
      if (!autoB2PreviewOutput) return;
      const m = (mode || "none").trim();
      const autoSingle = dbg.auto_b2_preview || {};
      const multi = dbg.auto_b2_multi || {};
      const multiPrev = dbg.auto_b2_multi_preview || {};
      const perLib = (dbg.libraries || []); // multi-library: per-library meta list

      function _perLibPreview(kind) {
        const out = {};
        (perLib || []).forEach((it) => {
          const lid = (it && (it.library_id || (it.debug || {}).library_id)) ? (it.library_id || (it.debug || {}).library_id) : "";
          const key = lid ? String(lid) : ("lib_" + String(Object.keys(out).length + 1));
          const d0 = (it && it.debug) ? it.debug : (it && it.debug === 0 ? {} : (it || {}));
          const pv = (d0 && d0.auto_b2_preview) ? d0.auto_b2_preview : {};
          if (kind === "dense") out[key] = pv.dense_queries || [];
          else if (kind === "sparse") out[key] = pv.sparse_terms || [];
          else if (kind === "glossary") out[key] = pv.glossary || {};
        });
        return out;
      }

      function _emptyHint() {
        const meta = dbg.auto_b2 || {};
        const meta2 = dbg.auto_b2_multi || {};
        return (
          "（当前无可显示的 Auto+B2 预览：可能未触发跨语检索，或该库尚未生成 language_profile 缓存。）\n\n" +
          "auto_b2:\n" + _fmtJson(meta) + "\n\n" +
          "auto_b2_multi:\n" + _fmtJson(meta2)
        );
      }
      if (m === "none") {
        autoB2PreviewOutput.textContent = "（不显示）";
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      if (m === "single_dense") {
        const arr = (autoSingle && autoSingle.dense_queries) ? autoSingle.dense_queries : null;
        if (arr && Array.isArray(arr) && arr.length) {
          autoB2PreviewOutput.textContent = "dense_queries:\n" + _fmtJson(arr);
        } else if (perLib && perLib.length) {
          autoB2PreviewOutput.textContent = "dense_queries（per library）:\n" + _fmtJson(_perLibPreview("dense"));
        } else {
          autoB2PreviewOutput.textContent = _emptyHint();
        }
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      if (m === "single_sparse") {
        const arr = (autoSingle && autoSingle.sparse_terms) ? autoSingle.sparse_terms : null;
        if (arr && Array.isArray(arr) && arr.length) {
          autoB2PreviewOutput.textContent = "sparse_terms:\n" + _fmtJson(arr);
        } else if (perLib && perLib.length) {
          autoB2PreviewOutput.textContent = "sparse_terms（per library）:\n" + _fmtJson(_perLibPreview("sparse"));
        } else {
          autoB2PreviewOutput.textContent = _emptyHint();
        }
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      if (m === "single_glossary") {
        const obj = (autoSingle && autoSingle.glossary) ? autoSingle.glossary : null;
        if (obj && Object.keys(obj || {}).length) {
          autoB2PreviewOutput.textContent = "glossary:\n" + _fmtJson(obj);
        } else if (perLib && perLib.length) {
          autoB2PreviewOutput.textContent = "glossary（per library）:\n" + _fmtJson(_perLibPreview("glossary"));
        } else {
          autoB2PreviewOutput.textContent = _emptyHint();
        }
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      if (m === "multi_groups") {
        autoB2PreviewOutput.textContent = "auto_b2_multi.groups:\n" + _fmtJson(multi.groups || {});
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      if (m === "multi_preview") {
        autoB2PreviewOutput.textContent = "auto_b2_multi_preview:\n" + _fmtJson(multiPrev || {});
        if (autoB2PreviewStatus) autoB2PreviewStatus.textContent = "";
        return;
      }
      autoB2PreviewOutput.textContent = "（未知模式）";
    }

    function parseKeywordTerms(raw) {
      if (!raw || !raw.trim()) return [];
      return raw
        .split(/[\n,，;；]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
    }

    /** 限定文件名：按行解析，不在英文逗号处切开（书名里常有 “Band, Hamburg” 等逗号）。 */
    function parseSourceFilters(raw) {
      if (!raw || !raw.trim()) return [];
      const out = [];
      for (const line of raw.split(/\r?\n/)) {
        const seg = line.trim();
        if (!seg) continue;
        for (const piece of seg.split(/[;；]/)) {
          const t = piece.trim();
          if (t) out.push(t);
        }
      }
      return out;
    }

    function applyStyleUIRules() {
      const style = ((answerStyleSelect && answerStyleSelect.value) ? answerStyleSelect.value : "").trim();
      const isStandardQA = style === "标准问答";
      const isSepMode = style === "斯坦福哲学百科模式";
      // 标准问答不依赖本地语料：禁用检索相关开关，避免用户误会
      autoKeywords.disabled = isStandardQA;
      useHybrid.disabled = isStandardQA;
      useRerank.disabled = isStandardQA;
      if (useSepReference) useSepReference.disabled = isStandardQA || isSepMode;
      if (isStandardQA) {
        autoKeywords.checked = false;
        useHybrid.checked = false;
        useRerank.checked = false;
        if (useSepReference) useSepReference.checked = false;
      }
      if (isSepMode && useSepReference) {
        // SEP 模式本身已经强制使用 sep profile，这个“参照”选项无意义
        useSepReference.checked = false;
      }
    }

    function setAskLoading(isLoading) {
      askButton.disabled = isLoading;
      if (!isLoading) {
        statusText.textContent = "";
        return;
      }
      const v = ((answerModelSelect && answerModelSelect.value) ? answerModelSelect.value : "gemini:gemini-3.1-pro-preview");
      if (v.startsWith("gemini:gemini-2.5-flash")) {
        statusText.textContent = "正在生成回答（Gemini: 2.5-flash）…";
      } else if (v.startsWith("gemini:gemini-2.5-pro")) {
        statusText.textContent = "正在生成回答（Gemini: 2.5-pro，单模型失败 3 次后切到 2.5-flash）…";
      } else if (v.startsWith("gemini:gemini-3.1-pro") || v.startsWith("gemini:gemini-3-pro")) {
        statusText.textContent = "正在生成回答（Gemini: 3.1-pro-preview，单模型失败 3 次后降级：2.5-pro → 2.5-flash）…";
      } else if (v.startsWith("gemini:")) {
        statusText.textContent = "正在生成回答（Gemini）…";
      } else if (v.startsWith("openai:")) {
        statusText.textContent = "正在生成回答（OpenAI）…";
      } else if (v.startsWith("deepseek:")) {
        statusText.textContent = "正在生成回答（DeepSeek）…";
      } else {
        statusText.textContent = "正在生成回答…";
      }
    }

    function setIngestLoading(isLoading) {
      ingestButton.disabled = isLoading;
      profileSelect.disabled = isLoading;
      if (uploadDropzone) {
        uploadDropzone.classList.toggle("disabled", isLoading);
        uploadDropzone.setAttribute("aria-disabled", isLoading ? "true" : "false");
      }
      if (uploadFileInput) uploadFileInput.disabled = isLoading;
    }

    function setDropzoneBusy(isBusy) {
      if (!uploadDropzone) return;
      uploadDropzone.classList.toggle("disabled", isBusy);
      uploadDropzone.setAttribute("aria-disabled", isBusy ? "true" : "false");
      if (uploadFileInput) uploadFileInput.disabled = isBusy;
    }

    const LAST_LIB_ID_KEY = "philo_last_ingest_library_id";

    function _getLastLibraryId() {
      try {
        const v = (localStorage.getItem(LAST_LIB_ID_KEY) || "").trim();
        return v || "default";
      } catch (e) {
        return "default";
      }
    }

    function _setLastLibraryId(v) {
      try {
        if (!v) return;
        localStorage.setItem(LAST_LIB_ID_KEY, String(v));
      } catch (e) {}
    }

    // Init library_id input from localStorage
    try {
      const initV = _getLastLibraryId();
      if (libraryIdInput && initV && initV !== "default") {
        libraryIdInput.value = initV;
      }
    } catch (e) {}

    function _getLibraryIdFromUI() {
      try {
        const v = libraryIdInput ? String(libraryIdInput.value || "").trim() : "";
        return v || "";
      } catch (e) {
        return "";
      }
    }

    function _setLibraryIdToUI(v) {
      try {
        if (!libraryIdInput) return;
        libraryIdInput.value = String(v || "");
      } catch (e) {}
    }

    function promptLibraryId(defaultValue) {
      // Prefer the visible input box; fallback to prompt only when empty.
      const ui = _getLibraryIdFromUI();
      if (ui) {
        _setLastLibraryId(ui);
        return ui;
      }
      const d = (defaultValue || _getLastLibraryId() || "default");
      const v = prompt("请输入本次库名（library_id）。支持空格，系统会自动转成下划线；留空=default", d);
      const out = (v == null ? "" : String(v)).trim() || "default";
      _setLastLibraryId(out);
      _setLibraryIdToUI(out === "default" ? "" : out);
      return out;
    }

    async function uploadFiles(fileList) {
      const allowed = /\.(pdf|docx|json|epub)$/i;
      const picked = Array.from(fileList).filter((f) => allowed.test(f.name));
      if (picked.length === 0) {
        ingestStatus.textContent = "请选择 .pdf / .docx / .json / .epub 文件。";
        return;
      }
      const library_id = promptLibraryId("default");
      const fd = new FormData();
      picked.forEach((f) => fd.append("files", f));
      fd.append("library_id", library_id);
      setDropzoneBusy(true);
      ingestStatus.textContent = `正在上传 ${picked.length} 个文件…`;
      try {
        const resp = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: fd });
        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(errText || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        const names = (data.saved || []).join("、");
        const savedDir = data.dir || "data/uploads";
        ingestStatus.textContent = `已保存 ${data.count || 0} 个文件到 ${savedDir}（library_id=${data.library_id || library_id}）：${names || "（无）"}`;
        if (autoIngestAfterUpload.checked) {
          await runIngestStream(library_id);
        }
      } catch (err) {
        console.error(err);
        ingestStatus.textContent = `上传失败：${(err && err.message) ? err.message : err}`;
      } finally {
        setDropzoneBusy(false);
      }
    }

    async function runIngestStream(libraryIdOverride) {
      setIngestLoading(true);
      ingestProgressWrap.classList.add("visible");
      ingestProgress.value = 0;
      ingestProgressLabel.textContent = "连接后端…";
      ingestStatus.textContent = "正在运行 ingest（流式进度）…";
      try {
        const library_id = (libraryIdOverride && String(libraryIdOverride).trim()) ? String(libraryIdOverride).trim() : promptLibraryId("default");
        const resp = await fetch(`${API_BASE}/api/ingest/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ library_id }),
        });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for (const line of lines) {
            if (!line.trim()) continue;
            let ev;
            try {
              ev = JSON.parse(line);
            } catch (e) {
              continue;
            }
            if (ev.type === "progress") {
              const pct = typeof ev.percent === "number" ? ev.percent : 0;
              ingestProgress.value = Math.min(100, Math.max(0, pct));
              ingestProgressLabel.textContent =
                ev.message || ev.stage || "";
            }
            if (ev.type === "done") {
              ingestProgress.value = 100;
              ingestProgressLabel.textContent = ev.message || "完成";
              ingestStatus.textContent = `Ingest 完成：共加载 ${ev.total_pages} 页，切分为 ${ev.total_chunks} 个语义片段。`;
            }
            if (ev.type === "error") {
              throw new Error(ev.message || "ingest 失败");
            }
          }
        }
      } catch (err) {
        console.error(err);
        ingestStatus.textContent = `Ingest 失败：${(err && err.message) ? err.message : err}。请检查后端与 data 目录。`;
        ingestProgressLabel.textContent = "";
      } finally {
        ingestButton.disabled = false;
        profileSelect.disabled = false;
        ingestProgressWrap.classList.remove("visible");
        if (uploadDropzone) {
          uploadDropzone.classList.remove("disabled");
          uploadDropzone.setAttribute("aria-disabled", "false");
        }
        if (uploadFileInput) uploadFileInput.disabled = false;
      }
    }

    async function loadProfile() {
      try {
        const resp = await fetch(`${API_BASE}/api/profile`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (Array.isArray(data.available_profiles) && data.available_profiles.length > 0) {
          profileSelect.innerHTML = "";
          data.available_profiles.forEach((p) => {
            const opt = document.createElement("option");
            opt.value = p;
            opt.textContent = p;
            profileSelect.appendChild(opt);
          });
        }
        profileSelect.value = data.profile || "quality";
      } catch (err) {
        console.error(err);
      }
    }

    async function switchProfile() {
      const profile = profileSelect.value;
      if (!profile) return;
      profileSelect.disabled = true;
      try {
        const resp = await fetch(`${API_BASE}/api/profile`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ profile }),
        });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const data = await resp.json();
        profileSelect.value = data.profile || profile;
        ingestStatus.textContent = `已切换到 ${data.profile}。请重新运行 Ingest 以重建向量与稀疏索引。`;
      } catch (err) {
        console.error(err);
        ingestStatus.textContent = "切换配置失败，请检查后端日志。";
      } finally {
        profileSelect.disabled = false;
      }
    }

    function renderAnswer(answer) {
      if (!answer) {
        answerContent.classList.add("placeholder");
        answerContent.textContent = "暂无回答。";
        return;
      }
      answerContent.classList.remove("placeholder");
      answerContent.textContent = answer;
    }

    function renderDocs(docs) {
      // Be defensive: if API returns unexpected shapes, avoid breaking the whole UI.
      currentDocs = Array.isArray(docs) ? docs : [];
      try {
        renderDocList(excerptList, currentDocs, "main");
      } catch (err) {
        console.error("renderDocs failed", err, docs);
        sidebarSubtitle.textContent =
          "侧边栏渲染失败：docs 数据结构异常或浏览器兼容性问题。请打开调试面板查看 debug，并在控制台查看错误。";
        excerptDetailCard.style.display = "none";
        return;
      }
      if (!currentDocs.length) {
        sidebarSubtitle.textContent = "未检索到任何文献片段。可以尝试换一个更宽泛或更具体的问题。";
        excerptDetailCard.style.display = "none";
        return;
      }
      sidebarSubtitle.textContent = `共检索到 ${currentDocs.length} 个文献片段。点击某一条查看全文。`;
    }

    function renderKeywordHitDocs(docs, sourceStats = []) {
      currentKeywordHitDocs = docs || [];
      renderDocList(keywordHitList, currentKeywordHitDocs, "keyword");
      if (!sourceStats || sourceStats.length === 0) {
        keywordSourceStats.textContent = "source 统计：暂无";
      } else {
        const top = sourceStats.slice(0, 12).map((x) => `${x.source}: ${x.count}`).join("\n");
        keywordSourceStats.textContent = `source 统计（Top 12）:\n${top}`;
      }
      if (!currentKeywordHitDocs.length) {
        keywordHitsSubtitle.textContent = "当前无关键词全量命中结果（可能未输入关键词或未命中）。";
        return;
      }
      keywordHitsSubtitle.textContent = `共命中 ${currentKeywordHitDocs.length} 个关键词相关 chunks。`;
    }

    function renderDocList(container, docs, kind) {
      container.innerHTML = "";
      docs.forEach((doc, idx) => {
        const item = document.createElement("div");
        item.className = "excerpt-item";
        item.dataset.index = idx;
        item.dataset.kind = kind;

        const meta = document.createElement("div");
        meta.className = "excerpt-meta";
        meta.textContent = `${(doc.source == null ? "Unknown" : doc.source)} (p. ${(doc.page == null ? "?" : doc.page)})`;

        const text = document.createElement("div");
        text.className = "excerpt-text";
        text.textContent = (doc.text == null ? "" : doc.text);

        item.appendChild(meta);
        item.appendChild(text);

        item.addEventListener("click", () => selectExcerpt(idx, kind));

        container.appendChild(item);
      });
    }

    function selectExcerpt(index, kind = "main") {
      const targetDocs = kind === "keyword" ? currentKeywordHitDocs : currentDocs;
      if (!targetDocs || index < 0 || index >= targetDocs.length) return;

      document.querySelectorAll(".excerpt-item").forEach((el) => {
        const sameKind = el.dataset.kind === kind;
        const sameIndex = Number(el.dataset.index) === index;
        if (sameKind && sameIndex) el.classList.add("active");
        else el.classList.remove("active");
      });

      const doc = targetDocs[index];
      excerptDetailCard.style.display = "block";
      excerptDetailTitle.textContent = `${(doc.source == null ? "Unknown" : doc.source)} (p. ${(doc.page == null ? "?" : doc.page)})`;
      excerptDetailText.textContent = (doc.text == null ? "" : doc.text);
      excerptDetailCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    async function runIngest() {
      await runIngestStream();
    }

    async function askQuestion() {
      const question = questionInput.value.trim();
      if (!question) {
        alert("请先输入一个问题。");
        return;
      }

      setAskLoading(true);
      renderAnswer("");
      if (answerModelBadge) answerModelBadge.textContent = "模型：生成中...";
      renderDocs([]);
      renderKeywordHitDocs([], []);
      renderDebug(null);

      try {
        setStatus("发送", "正在请求后端 /api/answer …");
        const keyword_terms = parseKeywordTerms(keywordInput.value);
        const source_filters = parseSourceFilters(sourceFilterInput.value);
        const rawModel = answerModelSelect.value || "gemini:gemini-3.1-pro-preview";
        const parts = rawModel.split(":");
        const llm_provider = parts[0] || "gemini";
        const llm_model = parts.slice(1).join(":") || (llm_provider === "gemini" ? "gemini-3.1-pro-preview" : "");
        const answer_style = answerStyleSelect.value || "哲学论述";
        const isStandardQA = (answer_style || "").trim() === "标准问答";
        const use_sep_reference = !!(useSepReference && useSepReference.checked);
        const conversation_id = getConversationId();
        const resp = await fetch(`${API_BASE}/api/answer`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question,
            user_id: USER_ID,
            conversation_id: conversation_id || null,
            memory: true,
            history_max_turns: 10,
            history_max_chars: 12000,
            wiki_max_chars: 3500,
            use_concept_graph: true,
            use_concept_index: false,
            keyword_terms: isStandardQA ? null : (keyword_terms.length ? keyword_terms : null),
            source_filters: isStandardQA ? null : (source_filters.length ? source_filters : null),
            auto_extract_keywords: isStandardQA ? false : autoKeywords.checked,
            use_hybrid: isStandardQA ? false : useHybrid.checked,
            use_rerank: isStandardQA ? false : useRerank.checked,
            use_sep_reference: isStandardQA ? false : use_sep_reference,
            ultra_long_answer: !!(ultraLongAnswer && ultraLongAnswer.checked),
            answer_style,
            llm_provider,
            llm_model,
            library_ids: getSelectedLibraries().map((x) => x.key),
            library_weights: getSelectedLibraries().map((x) => x.weight),
          }),
        });

        if (!resp.ok) {
          const errText = await resp.text().catch(() => "");
          throw new Error(errText || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        if (data && data.conversation_id) {
          setConversationId(data.conversation_id);
          ensureSessionExists(data.conversation_id);
          updateSessionUpdatedAt(data.conversation_id);
          renderSessionSelect();
        }
        renderAnswer(data.answer);
        if (answerModelBadge) {
          answerModelBadge.textContent = `模型：${data.answer_model || "unknown"}`;
        }
        renderDocs(data.docs);
        renderKeywordHitDocs(data.keyword_hit_docs || [], data.keyword_source_stats || []);
        renderDebug(data);
        const kw = data.keywords_used && data.keywords_used.length
          ? data.keywords_used.join("、")
          : (data.keyword_query || "").slice(0, 80);
        const isStd = (data.answer_style || "").trim() === "标准问答" || !!data.retrieval_skipped;
        if (isStd) {
          sidebarSubtitle.textContent = "标准问答模式：未使用本地语料/向量库，本栏不会展示文献片段。";
        } else {
          const hybridOn = data.hybrid ? "混合检索已生效" : "仅稠密检索（未启用混合或未构建 FTS5 索引）";
          const rerankOn = data.reranked ? "已启用重排序" : "未启用重排序";
          sidebarSubtitle.textContent =
            `共检索到 ${(data.docs || []).length} 个文献片段。${hybridOn}，${rerankOn}。稀疏查询：${kw || "（整句问题）"}`;
        }
        statusText.textContent = `本次回答模型：${data.answer_model || "unknown"}`;

        // Refresh memory/wiki viewer: wiki updates run in a background thread,
        // so we do an immediate refresh + a delayed refresh.
        try { await refreshMemoryViewer(); } catch (e2) {}
        try { setTimeout(() => { try { refreshMemoryViewer(); } catch (e3) {} }, 1200); } catch (e4) {}
        // Auto-name session (only if still default)
        try { setTimeout(() => { autoNameCurrentSession({ question, answer: data.answer, onlyIfDefault: true }); }, 50); } catch (e5) {}
      } catch (err) {
        console.error(err);
        renderAnswer("");
        if (answerModelBadge) answerModelBadge.textContent = "模型：失败";
        renderKeywordHitDocs([], []);
        renderDebug(null);
        answerContent.classList.add("placeholder");
        answerContent.textContent = "请求失败：请检查后端是否已启动（uvicorn），或看页面顶部状态提示。";
        setStatus("失败", (err && err.message) ? err.message : String(err));
      } finally {
        setAskLoading(false);
      }
    }

    ingestButton.addEventListener("click", runIngest);

    if (uploadDropzone && uploadFileInput) {
      // 关键：部分浏览器在页面未全局 preventDefault 时，会把文件拖拽当作“打开文件/导航”，
      // 导致 dropzone 的 drop 事件不稳定或用户感知为“拖不进去”。
      // 这里用捕获阶段拦截默认行为，但不影响 dropzone 内部处理。
      window.addEventListener(
        "dragover",
        (e) => {
          e.preventDefault();
        },
        true
      );
      window.addEventListener(
        "drop",
        (e) => {
          // 允许 dropzone 自己处理；其它位置 drop 则阻止浏览器打开文件
          if (!uploadDropzone.contains(e.target)) {
            e.preventDefault();
          }
        },
        true
      );

      uploadDropzone.addEventListener("click", () => {
        if (uploadDropzone.classList.contains("disabled")) return;
        uploadFileInput.click();
      });
      uploadDropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (!uploadDropzone.classList.contains("disabled")) uploadFileInput.click();
        }
      });
      uploadFileInput.addEventListener("change", () => {
        if (uploadFileInput.files && uploadFileInput.files.length) {
          uploadFiles(uploadFileInput.files);
          uploadFileInput.value = "";
        }
      });
      ["dragenter", "dragover", "dragleave", "drop"].forEach((evName) => {
        uploadDropzone.addEventListener(evName, (e) => {
          e.preventDefault();
          e.stopPropagation();
        });
      });
      uploadDropzone.addEventListener("dragover", () => {
        setStatus("上传", "检测到拖拽文件，松手即可上传…");
        if (!uploadDropzone.classList.contains("disabled")) uploadDropzone.classList.add("dragover");
      });
      uploadDropzone.addEventListener("dragleave", () => {
        uploadDropzone.classList.remove("dragover");
      });
      uploadDropzone.addEventListener("drop", (e) => {
        uploadDropzone.classList.remove("dragover");
        if (uploadDropzone.classList.contains("disabled")) return;
        const files = e.dataTransfer && e.dataTransfer.files;
        setStatus("上传", `已放下文件，准备上传…（count=${files ? files.length : 0}）`);
        if (files && files.length) uploadFiles(files);
      });
    }
    profileSelect.addEventListener("change", switchProfile);
    answerStyleSelect.addEventListener("change", applyStyleUIRules);
    // 提问按钮：只绑定一次，避免一次点击触发两次 /api/answer（双并发会显著拖慢响应）
    // 兼容性兜底交给 ES5 fallback；现代主脚本不再同时设置 onclick + addEventListener。
    if (askButton) {
      askButton.disabled = false;
      askButton.addEventListener("click", () => {
        // 若按钮已 disabled，则说明请求进行中；忽略重复点击（防双发）
        if (askButton.disabled) return;
        setStatus("点击", "已点击“提问”，准备校验输入…");
        askQuestion();
      });
    } else {
      setStatus("JS错误", "找不到 #askButton（前端 DOM 异常）");
    }
    loadProfile();
    applyStyleUIRules();
    if (refreshLibrariesBtn) {
      refreshLibrariesBtn.onclick = loadLibraries;
    }
    if (toggleLibraryPanelBtn && libraryPanel) {
      toggleLibraryPanelBtn.addEventListener("click", () => {
        const open = libraryPanel.style.display === "none";
        libraryPanel.style.display = open ? "block" : "none";
        if (open) {
          loadLibraries();
          loadTrashItems();
        }
      });
    }
    if (detectLibraryLangBtn) {
      detectLibraryLangBtn.onclick = function () {
        detectSelectedLibrariesLanguage();
      };
    }
    if (loadSourcesBtn) {
      loadSourcesBtn.onclick = function () {
        setStatus("点击", "已点击「加载」文献列表…");
        loadLibrarySources();
      };
    }
    {
      const lss0 = libSourcesSel();
      if (lss0) {
        lss0.addEventListener("change", () => {
          loadLibrarySources();
        });
      }
    }
    if (refreshTrashBtn) {
      refreshTrashBtn.onclick = loadTrashItems;
    }
    if (refreshMemoryBtn) {
      refreshMemoryBtn.onclick = refreshMemoryViewer;
    }
    if (sessionSelect) {
      sessionSelect.onchange = () => {
        const v = String(sessionSelect.value || "").trim();
        if (v) switchSession(v);
      };
    }
    if (newSessionBtn) {
      newSessionBtn.onclick = () => {
        const id = _newConvId();
        const list = _loadSessions();
        list.unshift({ id, title: "新会话", updatedAt: _nowIso() });
        _saveSessions(list.slice(0, 80));
        switchSession(id);
      };
    }
    if (autoNameSessionBtn) {
      autoNameSessionBtn.onclick = async () => {
        const q = (questionInput && questionInput.value) ? String(questionInput.value).trim() : "";
        const a = (answerContent && answerContent.textContent) ? String(answerContent.textContent).trim() : "";
        await autoNameCurrentSession({ question: q, answer: a, onlyIfDefault: false });
        renderSessionSelect();
      };
    }
    if (renameSessionBtn) {
      renameSessionBtn.onclick = () => {
        const cur = getConversationId();
        if (!cur) return;
        const title = prompt("会话名称：", "新会话");
        if (title == null) return;
        const t = String(title || "").trim();
        if (t) setSessionTitle(cur, t);
      };
    }
    if (deleteSessionBtn) {
      deleteSessionBtn.onclick = () => {
        const cur = getConversationId();
        if (!cur) return;
        const ok = confirm("删除当前会话（仅从列表移除，不会立即删除磁盘历史文件）？");
        if (!ok) return;
        const list = _loadSessions().filter((it) => it && String(it.id || "") !== cur);
        _saveSessions(list.slice(0, 80));
        // switch to first remaining or create a new one
        const next = (list[0] && list[0].id) ? String(list[0].id) : "";
        if (next) {
          switchSession(next);
        } else {
          const id = _newConvId();
          _saveSessions([{ id, title: "新会话", updatedAt: _nowIso() }]);
          switchSession(id);
        }
      };
    }
    const _tl = document.getElementById("toggleLeftbarBtn");
    if (_tl) _tl.onclick = toggleLeftbar;
    const _tr = document.getElementById("toggleRightbarBtn");
    if (_tr) _tr.onclick = toggleRightbar;
    if (compactWikiBtn) {
      compactWikiBtn.onclick = compactWikiNow;
    }
    if (toggleMemoryHistoryBtn && memoryHistoryPanel) {
      toggleMemoryHistoryBtn.addEventListener("click", () => {
        const open = memoryHistoryPanel.style.display === "none";
        memoryHistoryPanel.style.display = open ? "block" : "none";
      });
    }
    loadLibraries();
    // init session selector
    try {
      const cur = getConversationId();
      if (cur) ensureSessionExists(cur);
      else {
        const id = _newConvId();
        setConversationId(id);
        ensureSessionExists(id);
      }
      // Merge in old sessions from disk (if any), then render.
      seedSessionsFromDisk().then(() => {
        try { renderSessionSelect(); } catch (e2) {}
      });
      renderSessionSelect();
    } catch (e) {}
    try { applySidebarState(); } catch (e) {}
    refreshMemoryViewer();
    statusText.insertAdjacentHTML(
      "beforebegin",
      '<button id="toggleDebugButton" class="btn secondary" style="margin-right:8px;">显示调试面板</button>'
    );
    const toggleDebugButton = document.getElementById("toggleDebugButton");
    toggleDebugButton.addEventListener("click", () => {
      const hidden = debugBar.classList.toggle("hidden");
      toggleDebugButton.textContent = hidden ? "显示调试面板" : "隐藏调试面板";
    });
    if (autoB2PreviewSelect) {
      autoB2PreviewSelect.addEventListener("change", () => {
        const dbg = (window.__LAST_DEBUG__ || {});
        updateAutoB2Preview(autoB2PreviewSelect.value, dbg);
      });
    }
    questionInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        askQuestion();
      }
    });
    window.__MAIN_SCRIPT_OK__ = true;
    } catch (e) {
      try {
        window.__MAIN_SCRIPT_ERROR__ = (e && (e.stack || e.message)) ? String(e.stack || e.message) : String(e);
      } catch (e2) {
        window.__MAIN_SCRIPT_ERROR__ = "unknown";
      }
      try {
        var b = document.getElementById("jsCompatBanner");
        if (b) b.textContent = "主脚本运行期错误：" + window.__MAIN_SCRIPT_ERROR__;
      } catch (e3) {}
    }

})();
