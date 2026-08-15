"use strict";

const elements = {
  status: document.querySelector("#system-status"),
  documentCount: document.querySelector("#document-count"),
  documentList: document.querySelector("#document-list"),
  uploadForm: document.querySelector("#upload-form"),
  fileInput: document.querySelector("#file-input"),
  uploadProgress: document.querySelector("#upload-progress"),
  uploadProgressDetail: document.querySelector("#upload-progress-detail"),
  queryForm: document.querySelector("#query-form"),
  questionInput: document.querySelector("#question-input"),
  askButton: document.querySelector("#ask-button"),
  queryProgress: document.querySelector("#query-progress"),
  resultSection: document.querySelector("#result-section"),
  emptyState: document.querySelector("#empty-state"),
  answerCard: document.querySelector("#answer-card"),
  answerState: document.querySelector("#answer-state"),
  answerTiming: document.querySelector("#answer-timing"),
  answerText: document.querySelector("#answer-text"),
  answerCitations: document.querySelector("#answer-citations"),
  evidenceCount: document.querySelector("#evidence-count"),
  evidenceList: document.querySelector("#evidence-list"),
  retrievalTrace: document.querySelector("#retrieval-trace"),
  toast: document.querySelector("#toast"),
};

let toastTimer;
const COLLAPSIBLE_EXCERPT_CHARACTERS = 420;
const COLLAPSIBLE_TRACE_CHARACTERS = 180;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = response.headers.get("content-type")?.includes("application/json");
  const body = isJson ? await response.json() : null;
  if (!response.ok) {
    throw new Error(body?.detail || `Request failed with status ${response.status}`);
  }
  return body;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  toastTimer = window.setTimeout(() => elements.toast.classList.add("hidden"), 5000);
}

function setSystemStatus(ready, message) {
  elements.status.replaceChildren();
  const dot = document.createElement("span");
  dot.className = `status-dot${ready ? "" : " status-dot--error"}`;
  const text = document.createElement("span");
  text.textContent = message;
  elements.status.append(dot, text);
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    if (status.ready) {
      setSystemStatus(true, "Offline models ready");
    } else {
      const missing = [];
      if (!status.database.ready) missing.push("database");
      if (!status.ollama.embedding_ready) missing.push("embedding model");
      if (!status.ollama.generation_ready) missing.push("generation model");
      setSystemStatus(false, `Needs ${missing.join(", ") || "local service"}`);
    }
  } catch (error) {
    setSystemStatus(false, "Local service unavailable");
    showToast(error.message);
  }
}

function fileTypeLabel(mediaType) {
  if (mediaType === "application/pdf") return "PDF";
  if (mediaType.startsWith("image/")) return "IMG";
  if (mediaType === "text/markdown") return "MD";
  return "TXT";
}

function renderDocuments(documents) {
  elements.documentCount.textContent = `${documents.length} / 20`;
  elements.documentList.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "document-empty";
    empty.textContent = "No sources indexed yet.";
    elements.documentList.append(empty);
    return;
  }

  for (const documentItem of documents) {
    const card = document.createElement("article");
    card.className = "document-card";
    const icon = document.createElement("span");
    icon.className = "file-icon";
    icon.textContent = fileTypeLabel(documentItem.media_type);
    const metadata = document.createElement("div");
    metadata.className = "document-meta";
    const name = document.createElement("strong");
    name.title = documentItem.filename;
    name.textContent = documentItem.filename;
    const detail = document.createElement("small");
    const suffix = documentItem.chunk_count === 1 ? "" : "s";
    detail.textContent = `${documentItem.chunk_count} chunk${suffix}`;
    metadata.append(name, detail);
    const remove = document.createElement("button");
    remove.className = "icon-button";
    remove.type = "button";
    remove.title = `Delete ${documentItem.filename}`;
    remove.setAttribute("aria-label", `Delete ${documentItem.filename}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => deleteDocument(documentItem));
    card.append(icon, metadata, remove);
    elements.documentList.append(card);
  }
}

async function refreshDocuments() {
  try {
    const result = await api("/api/documents");
    renderDocuments(result.documents);
  } catch (error) {
    showToast(error.message);
  }
}

async function uploadFile(file) {
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  elements.uploadProgressDetail.textContent = `Processing ${file.name}…`;
  elements.uploadProgress.classList.remove("hidden");
  elements.fileInput.disabled = true;
  try {
    await api("/api/documents", { method: "POST", body: formData });
    elements.fileInput.value = "";
    await refreshDocuments();
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.fileInput.disabled = false;
    elements.uploadProgress.classList.add("hidden");
  }
}

async function deleteDocument(documentItem) {
  const confirmed = window.confirm(`Delete “${documentItem.filename}” from this local vault?`);
  if (!confirmed) return;
  try {
    await api(`/api/documents/${documentItem.id}`, { method: "DELETE" });
    await refreshDocuments();
  } catch (error) {
    showToast(error.message);
  }
}

function rankChip(text) {
  const chip = document.createElement("span");
  chip.className = "rank-chip";
  chip.textContent = text;
  return chip;
}

function renderEvidence(evidence) {
  elements.evidenceList.replaceChildren();
  const suffix = evidence.length === 1 ? "" : "s";
  elements.evidenceCount.textContent = `${evidence.length} source${suffix}`;
  if (!evidence.length) {
    const empty = document.createElement("p");
    empty.className = "document-empty";
    empty.textContent = "No evidence passed the retrieval threshold.";
    elements.evidenceList.append(empty);
    return;
  }

  evidence.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "evidence-card";
    const topline = document.createElement("div");
    topline.className = "evidence-topline";
    const source = document.createElement("div");
    const label = document.createElement("span");
    label.className = "source-label";
    label.textContent = `[${item.label}]`;
    const name = document.createElement("span");
    name.className = "source-name";
    name.textContent = item.filename;
    source.append(label, name);
    if (item.page_number) {
      const page = document.createElement("span");
      page.className = "source-page";
      page.textContent = `· page ${item.page_number}`;
      source.append(page);
    }
    const modality = document.createElement("span");
    modality.className = "source-modality";
    modality.textContent = item.modality;
    topline.append(source, modality);
    const excerpt = document.createElement("p");
    excerpt.className = "evidence-excerpt";
    excerpt.id = `evidence-excerpt-${item.chunk_id}-${index}`;
    excerpt.textContent = item.excerpt;
    const isExpandable = item.excerpt.length > COLLAPSIBLE_EXCERPT_CHARACTERS;
    let toggle;
    if (isExpandable) {
      excerpt.classList.add("evidence-excerpt--collapsed");
      toggle = document.createElement("button");
      toggle.className = "excerpt-toggle";
      toggle.type = "button";
      toggle.setAttribute("aria-controls", excerpt.id);
      toggle.setAttribute("aria-expanded", "false");
      toggle.textContent = "Expand full chunk";
      toggle.addEventListener("click", () => {
        const isExpanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", String(!isExpanded));
        toggle.textContent = isExpanded ? "Expand full chunk" : "Collapse chunk";
        excerpt.classList.toggle("evidence-excerpt--collapsed", isExpanded);
      });
    }
    const ranks = document.createElement("div");
    ranks.className = "rank-row";
    if (item.lexical_rank) ranks.append(rankChip(`Keyword #${item.lexical_rank}`));
    if (item.semantic_rank) {
      const score = item.semantic_score.toFixed(3);
      ranks.append(rankChip(`Semantic #${item.semantic_rank} · ${score}`));
    }
    ranks.append(rankChip(`RRF ${item.fused_score.toFixed(4)}`));
    card.append(topline, excerpt);
    if (toggle) card.append(toggle);
    card.append(ranks);
    elements.evidenceList.append(card);
  });
}

function renderTrace(trace) {
  elements.retrievalTrace.replaceChildren();
  if (!trace.length) {
    const empty = document.createElement("p");
    empty.className = "document-empty";
    empty.textContent = "No chunks ranked for this question.";
    elements.retrievalTrace.append(empty);
    return;
  }

  trace.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "trace-row";
    const header = document.createElement("div");
    header.className = "trace-row-header";
    const position = document.createElement("span");
    position.className = "trace-position";
    position.textContent = `#${index + 1}`;
    const name = document.createElement("span");
    name.className = "trace-name";
    const page = item.page_number ? ` · p.${item.page_number}` : "";
    name.textContent = `${item.filename}${page}`;
    const scores = document.createElement("span");
    scores.className = "trace-scores";
    const lexical = item.lexical_rank ? `K${item.lexical_rank}` : "K—";
    const semantic = item.semantic_rank ? `S${item.semantic_rank}` : "S—";
    scores.textContent = `${lexical} · ${semantic} · ${item.fused_score.toFixed(4)}`;
    header.append(position, name, scores);

    const excerpt = document.createElement("p");
    excerpt.className = "trace-excerpt";
    excerpt.id = `trace-excerpt-${item.chunk_id}-${index}`;
    excerpt.textContent = item.excerpt;
    row.append(header, excerpt);

    if (item.excerpt.length > COLLAPSIBLE_TRACE_CHARACTERS) {
      excerpt.classList.add("trace-excerpt--collapsed");
      const toggle = document.createElement("button");
      toggle.className = "trace-toggle";
      toggle.type = "button";
      toggle.setAttribute("aria-controls", excerpt.id);
      toggle.setAttribute("aria-expanded", "false");
      toggle.textContent = "Expand chunk";
      toggle.addEventListener("click", () => {
        const isExpanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", String(!isExpanded));
        toggle.textContent = isExpanded ? "Expand chunk" : "Collapse chunk";
        excerpt.classList.toggle("trace-excerpt--collapsed", isExpanded);
      });
      row.append(toggle);
    }

    elements.retrievalTrace.append(row);
  });
}

function renderResult(result) {
  elements.emptyState.classList.add("hidden");
  elements.resultSection.classList.remove("hidden");
  elements.answerCard.classList.toggle("is-insufficient", !result.sufficient);
  elements.answerState.textContent = result.sufficient ? "Grounded answer" : "Insufficient evidence";
  elements.answerTiming.textContent = `${(result.timing.total_ms / 1000).toFixed(2)}s locally`;
  elements.answerText.textContent = result.answer;
  elements.answerCitations.replaceChildren();
  for (const citation of result.citations) {
    const chip = document.createElement("span");
    chip.className = "citation-chip";
    chip.textContent = `[${citation}]`;
    elements.answerCitations.append(chip);
  }
  renderEvidence(result.evidence);
  renderTrace(result.retrieval_trace);
  elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function askQuestion(question) {
  elements.askButton.disabled = true;
  elements.queryProgress.classList.remove("hidden");
  try {
    const result = await api("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderResult(result);
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.askButton.disabled = false;
    elements.queryProgress.classList.add("hidden");
  }
}

elements.uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  uploadFile(elements.fileInput.files[0]);
});

elements.fileInput.addEventListener("change", () => uploadFile(elements.fileInput.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  elements.uploadForm.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadForm.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.uploadForm.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadForm.classList.remove("is-dragging");
  });
}

elements.uploadForm.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));

elements.queryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  if (question) askQuestion(question);
});

elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.queryForm.requestSubmit();
  }
});

Promise.all([refreshStatus(), refreshDocuments()]);
