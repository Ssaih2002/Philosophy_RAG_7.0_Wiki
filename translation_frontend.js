;(function () {
  "use strict";

  const API_BASE = "http://127.0.0.1:8000";
  const CURRENT_PROJECT_KEY = "philo_translation_current_project_id";

  const els = {
    apiBadge: document.getElementById("apiBadge"),
    fileInput: document.getElementById("translationFileInput"),
    targetInput: document.getElementById("translationTargetInput"),
    modelSelect: document.getElementById("translationModelSelect"),
    formatSelect: document.getElementById("translationFormatSelect"),
    projectSelect: document.getElementById("projectSelect"),
    maxChunksInput: document.getElementById("maxChunksInput"),
    concurrencyInput: document.getElementById("concurrencyInput"),
    prepareBtn: document.getElementById("translationPrepareBtn"),
    confirmBtn: document.getElementById("translationConfirmBtn"),
    runBtn: document.getElementById("translationRunBtn"),
    exportBtn: document.getElementById("translationExportBtn"),
    oneClickBtn: document.getElementById("translationOneClickBtn"),
    refreshBtn: document.getElementById("refreshProjectBtn"),
    loadGlobalGlossaryBtn: document.getElementById("translationLoadGlobalGlossaryBtn"),
    saveGlobalGlossaryBtn: document.getElementById("translationSaveGlobalGlossaryBtn"),
    status: document.getElementById("translationStatus"),
    glossaryInput: document.getElementById("translationGlossaryInput"),
    overview: document.getElementById("translationOverview"),
    eventLog: document.getElementById("eventLog"),
  };

  let currentProjectId = loadCurrentProjectId();
  let isBusy = false;

  function loadCurrentProjectId() {
    try {
      return (localStorage.getItem(CURRENT_PROJECT_KEY) || "").trim();
    } catch (e) {
      return "";
    }
  }

  function saveCurrentProjectId(projectId) {
    currentProjectId = String(projectId || "").trim();
    try {
      if (currentProjectId) localStorage.setItem(CURRENT_PROJECT_KEY, currentProjectId);
      else localStorage.removeItem(CURRENT_PROJECT_KEY);
    } catch (e) {}
  }

  function parseProviderModel(raw) {
    const value = String(raw || "gemini:gemini-3.1-pro-preview");
    const parts = value.split(":");
    return {
      provider: parts[0] || "gemini",
      model: parts.slice(1).join(":") || "",
    };
  }

  async function readJsonResponse(resp) {
    const text = await resp.text();
    let data = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (e) {
        throw new Error(text.slice(0, 500));
      }
    }
    if (!resp.ok) {
      throw new Error((data && data.detail) ? data.detail : `HTTP ${resp.status}`);
    }
    return data;
  }

  function setBusy(nextBusy) {
    isBusy = !!nextBusy;
    [
      els.prepareBtn,
      els.confirmBtn,
      els.runBtn,
      els.exportBtn,
      els.oneClickBtn,
      els.refreshBtn,
      els.loadGlobalGlossaryBtn,
      els.saveGlobalGlossaryBtn,
      els.fileInput,
      els.targetInput,
      els.modelSelect,
      els.formatSelect,
      els.projectSelect,
      els.maxChunksInput,
      els.concurrencyInput,
    ].forEach((el) => {
      if (el) el.disabled = isBusy;
    });
  }

  function setStatus(text, isError) {
    if (!els.status) return;
    els.status.textContent = text || "";
    els.status.classList.toggle("error", !!isError);
  }

  function appendEvents(events) {
    if (!els.eventLog || !Array.isArray(events) || !events.length) return;
    const lines = events.map((ev) => {
      const type = ev && ev.type ? ev.type : "event";
      return `[${new Date().toLocaleTimeString()}] ${type} ${JSON.stringify(ev, null, 0)}`;
    });
    const previous = els.eventLog.textContent && els.eventLog.textContent !== "事件日志：暂无"
      ? els.eventLog.textContent
      : "";
    els.eventLog.textContent = [previous, ...lines].filter(Boolean).join("\n");
    els.eventLog.scrollTop = els.eventLog.scrollHeight;
  }

  function renderProject(state, glossary) {
    if (!state) return;
    saveCurrentProjectId(state.project_id || "");
    const progress = state.progress || {};
    const overview = state.overview || {};
    if (els.overview) {
      els.overview.textContent = JSON.stringify({
        project_id: state.project_id,
        source_name: state.source_name,
        target_language: state.target_language,
        status: state.status,
        progress,
        overview: overview.overview || "",
        structure: overview.structure || [],
        last_export: state.last_export || "",
      }, null, 2);
    }
    if (els.glossaryInput && glossary) {
      els.glossaryInput.value = JSON.stringify(glossary, null, 2);
    }
    setStatus(`项目 ${currentProjectId || "-"}：${state.status || "-"}（${progress.translated_chunks || 0}/${progress.total_chunks || 0}）`);
  }

  function getGlossaryFromEditor() {
    try {
      return JSON.parse((els.glossaryInput && els.glossaryInput.value) ? els.glossaryInput.value : "{}");
    } catch (e) {
      throw new Error("术语表不是合法 JSON。");
    }
  }

  function getMaxChunks() {
    const raw = els.maxChunksInput ? Number(els.maxChunksInput.value || 0) : 0;
    return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 0;
  }

  function getConcurrency() {
    const raw = els.concurrencyInput ? Number(els.concurrencyInput.value || 1) : 1;
    if (!Number.isFinite(raw)) return 1;
    return Math.max(1, Math.min(20, Math.floor(raw)));
  }

  async function loadProjectList() {
    const resp = await fetch(`${API_BASE}/api/translation/projects`);
    const data = await readJsonResponse(resp);
    const projects = Array.isArray(data.projects) ? data.projects : [];
    if (!els.projectSelect) return projects;

    els.projectSelect.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = projects.length ? "选择已有项目" : "暂无项目";
    els.projectSelect.appendChild(empty);

    projects.forEach((project) => {
      const option = document.createElement("option");
      const progress = project.progress || {};
      option.value = project.project_id || "";
      option.textContent = `${project.project_id || "-"} | ${project.status || "-"} | ${progress.translated_chunks || 0}/${progress.total_chunks || 0} | ${project.source_name || ""}`;
      els.projectSelect.appendChild(option);
    });

    if (currentProjectId) {
      els.projectSelect.value = currentProjectId;
    }
    return projects;
  }

  async function loadSelectedProject(projectId) {
    const id = String(projectId || currentProjectId || "").trim();
    if (!id) {
      setStatus("请选择或创建一个翻译项目。", true);
      return null;
    }
    const resp = await fetch(`${API_BASE}/api/translation/projects/${encodeURIComponent(id)}`);
    const data = await readJsonResponse(resp);
    renderProject(data.state, data.glossary || data.glossary_draft);
    return data;
  }

  async function prepareProject() {
    if (!els.fileInput || !els.fileInput.files || !els.fileInput.files.length) {
      throw new Error("请先选择要翻译的 PDF / DOCX / JSON / EPUB。");
    }
    const pm = parseProviderModel(els.modelSelect && els.modelSelect.value);
    const form = new FormData();
    form.append("file", els.fileInput.files[0]);
    form.append("target_language", (els.targetInput && els.targetInput.value) ? els.targetInput.value.trim() : "zh-CN");
    form.append("llm_provider", pm.provider);
    form.append("llm_model", pm.model);

    setStatus("正在生成概览与术语表草稿，长文可能需要等待数分钟...");
    const resp = await fetch(`${API_BASE}/api/translation/prepare`, { method: "POST", body: form });
    const data = await readJsonResponse(resp);
    appendEvents(data.events);
    renderProject(data.state, data.glossary_draft);
    await loadProjectList();
    return data;
  }

  async function confirmGlossary() {
    if (!currentProjectId) {
      throw new Error("请先生成或选择翻译项目。");
    }
    const glossary = getGlossaryFromEditor();
    setStatus("正在确认术语表...");
    const resp = await fetch(`${API_BASE}/api/translation/projects/${encodeURIComponent(currentProjectId)}/glossary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ glossary }),
    });
    const data = await readJsonResponse(resp);
    if (els.glossaryInput) {
      els.glossaryInput.value = JSON.stringify(data.glossary || {}, null, 2);
    }
    setStatus("术语表已确认，可以开始翻译。");
    return data;
  }

  async function runTranslation() {
    if (!currentProjectId) {
      throw new Error("请先生成或选择翻译项目。");
    }
    const pm = parseProviderModel(els.modelSelect && els.modelSelect.value);
    setStatus("正在逐块翻译；长书可能需要较长时间，请保持后端运行...");
    const resp = await fetch(`${API_BASE}/api/translation/projects/${encodeURIComponent(currentProjectId)}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        llm_provider: pm.provider,
        llm_model: pm.model,
        max_chunks: getMaxChunks(),
        concurrency: getConcurrency(),
        resume: true,
      }),
    });
    const data = await readJsonResponse(resp);
    appendEvents(data.events);
    renderProject(data.state, null);
    await loadProjectList();
    return data;
  }

  async function exportTranslation() {
    if (!currentProjectId) {
      throw new Error("请先生成或选择翻译项目。");
    }
    const format = (els.formatSelect && els.formatSelect.value) ? els.formatSelect.value : "txt";
    setStatus("正在导出...");
    const resp = await fetch(`${API_BASE}/api/translation/projects/${encodeURIComponent(currentProjectId)}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format }),
    });
    const data = await readJsonResponse(resp);
    if (data.download_url) {
      window.open(`${API_BASE}${data.download_url}`, "_blank");
    }
    await loadSelectedProject(currentProjectId);
    setStatus(`导出完成：${data.path || ""}`);
    return data;
  }

  async function loadGlobalGlossary() {
    const target = (els.targetInput && els.targetInput.value) ? els.targetInput.value.trim() : "";
    const url = `${API_BASE}/api/translation/glossary` + (target ? `?target_language=${encodeURIComponent(target)}` : "");
    setStatus("正在加载长期术语库...");
    const resp = await fetch(url);
    const data = await readJsonResponse(resp);
    if (els.glossaryInput) {
      els.glossaryInput.value = JSON.stringify(data.glossary || {}, null, 2);
    }
    setStatus("长期术语库已加载到编辑框；修改后可点击“保存长期术语库”。");
  }

  async function saveGlobalGlossary() {
    const glossary = getGlossaryFromEditor();
    if (!confirm("确认用当前编辑框内容覆盖长期术语库？建议先确认这是 global_glossary 格式。")) {
      return;
    }
    setStatus("正在保存长期术语库...");
    const resp = await fetch(`${API_BASE}/api/translation/glossary`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ glossary }),
    });
    const data = await readJsonResponse(resp);
    if (els.glossaryInput) {
      els.glossaryInput.value = JSON.stringify(data.glossary || {}, null, 2);
    }
    setStatus(`长期术语库已保存：${((data.glossary || {}).terms || []).length} 条。`);
  }

  async function runGuarded(task, label) {
    if (isBusy) return;
    setBusy(true);
    try {
      await task();
    } catch (e) {
      setStatus(`${label}失败：${e && e.message ? e.message : String(e)}`, true);
    } finally {
      setBusy(false);
    }
  }

  async function checkHealth() {
    try {
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 2500);
      const resp = await fetch(`${API_BASE}/api/health`, {
        cache: "no-store",
        signal: controller.signal,
      });
      window.clearTimeout(timer);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (els.apiBadge) {
        els.apiBadge.textContent = "API: 已连接";
      }
      return true;
    } catch (e) {
      if (els.apiBadge) {
        els.apiBadge.textContent = "API: 未连接";
      }
      setStatus("后端 API 未连接。请先运行 start_translation.bat 或确认 8000 端口后端已启动。", true);
      return false;
    }
  }

  async function waitForBackend() {
    for (let attempt = 1; attempt <= 8; attempt += 1) {
      if (attempt > 1) {
        setStatus(`正在等待后端启动...（${attempt}/8）`);
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
      }
      if (await checkHealth()) {
        return true;
      }
    }
    return false;
  }

  function bindEvents() {
    if (els.prepareBtn) {
      els.prepareBtn.addEventListener("click", () => runGuarded(prepareProject, "生成概览与术语表"));
    }
    if (els.confirmBtn) {
      els.confirmBtn.addEventListener("click", () => runGuarded(confirmGlossary, "确认术语表"));
    }
    if (els.runBtn) {
      els.runBtn.addEventListener("click", () => runGuarded(runTranslation, "翻译"));
    }
    if (els.exportBtn) {
      els.exportBtn.addEventListener("click", () => runGuarded(exportTranslation, "导出"));
    }
    if (els.oneClickBtn) {
      els.oneClickBtn.addEventListener("click", () => runGuarded(async () => {
        await prepareProject();
        await confirmGlossary();
        await runTranslation();
        await exportTranslation();
      }, "一键翻译"));
    }
    if (els.refreshBtn) {
      els.refreshBtn.addEventListener("click", () => runGuarded(async () => {
        await loadProjectList();
        if (currentProjectId) await loadSelectedProject(currentProjectId);
      }, "刷新项目"));
    }
    if (els.loadGlobalGlossaryBtn) {
      els.loadGlobalGlossaryBtn.addEventListener("click", () => runGuarded(loadGlobalGlossary, "加载长期术语库"));
    }
    if (els.saveGlobalGlossaryBtn) {
      els.saveGlobalGlossaryBtn.addEventListener("click", () => runGuarded(saveGlobalGlossary, "保存长期术语库"));
    }
    if (els.projectSelect) {
      els.projectSelect.addEventListener("change", () => runGuarded(async () => {
        saveCurrentProjectId(els.projectSelect.value || "");
        if (currentProjectId) await loadSelectedProject(currentProjectId);
      }, "载入项目"));
    }
  }

  async function init() {
    bindEvents();
    const backendReady = await waitForBackend();
    if (!backendReady) {
      if (els.projectSelect) {
        els.projectSelect.innerHTML = '<option value="">后端未连接</option>';
      }
      setStatus("初始化暂停：无法连接后端 API。请确认后端窗口没有报错，并能打开 http://127.0.0.1:8000/api/health。", true);
      return;
    }
    try {
      await loadProjectList();
      if (currentProjectId) {
        await loadSelectedProject(currentProjectId);
      }
    } catch (e) {
      setStatus(`初始化失败：${e && e.message ? e.message : String(e)}`, true);
    }
  }

  init();
})();
