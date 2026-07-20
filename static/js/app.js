/* ═══════════════════════════════════════════════════════
   Nexus — Single-Page App JavaScript
   ═══════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────

const state = {
  sessions: [],
  activeSessionId: null,
  messages: [],
  documents: [],
  isTyping: false,
  apiConnected: false,
};

// ── DOM References ─────────────────────────────────

const el = {
  sidebar:            document.getElementById('sidebar'),
  sidebarOverlay:     document.getElementById('sidebarOverlay'),
  menuToggle:         document.getElementById('menuToggle'),
  sessionList:        document.getElementById('sessionList'),
  documentList:       document.getElementById('documentList'),
  emptyDocs:          document.getElementById('emptyDocs'),
  emptyState:         document.getElementById('emptyState'),
  emptyNewChatBtn:    document.getElementById('emptyNewChatBtn'),
  newChatBtn:         document.getElementById('newChatBtn'),
  chatView:           document.getElementById('chatView'),
  messagesList:       document.getElementById('messagesList'),
  messagesContainer:  document.getElementById('messagesContainer'),
  typingIndicator:    document.getElementById('typingIndicator'),
  messageInput:       document.getElementById('messageInput'),
  sendBtn:            document.getElementById('sendBtn'),
  fileInput:          document.getElementById('fileInput'),
  fileInput2:         document.getElementById('fileInput2'),
  toastContainer:     document.getElementById('toastContainer'),
  connectionDot:      document.getElementById('connectionDot'),
  connectionText:     document.getElementById('connectionText'),
  confirmOverlay:     document.getElementById('confirmOverlay'),
  confirmMessage:     document.getElementById('confirmMessage'),
  confirmYes:         document.getElementById('confirmYes'),
  confirmNo:          document.getElementById('confirmNo'),
};

// ── API Client ─────────────────────────────────────

async function api(path, options) {
  options = options || {};

  // For FormData uploads, don't set Content-Type — let the browser
  // auto-set it to multipart/form-data with the correct boundary.
  if (options.body instanceof FormData) {
    const res = await fetch(path, options);
    return _handleResponse(res);
  }

  const config = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };

  const res = await fetch(path, config);
  return _handleResponse(res);
}

async function _handleResponse(res) {
  if (!res.ok) {
    let detail = 'HTTP ' + res.status;
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}
api.get    = function (path)           { return api(path); };
api.post   = function (path, data)     { return api(path, { method: 'POST',   body: JSON.stringify(data) }); };
api.patch  = function (path, data)     { return api(path, { method: 'PATCH',  body: JSON.stringify(data) }); };
api.delete = function (path)           { return api(path, { method: 'DELETE' }); };
api.upload = function (path, formData) { return api(path, { method: 'POST',   body: formData }); };

// ── Toast System ───────────────────────────────────

function showToast(message, type, duration) {
  type = type || 'info';
  duration = duration || 4000;
  var icons = { success: '✓', error: '✕', info: '●' };
  var toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.innerHTML =
    '<span class="toast-icon">' + (icons[type] || '●') + '</span>' +
    '<span class="toast-text">' + escapeHtml(message) + '</span>' +
    '<span class="toast-close">✕</span>';

  toast.querySelector('.toast-close').addEventListener('click', function () {
    dismissToast(toast);
  });

  el.toastContainer.appendChild(toast);

  if (duration > 0) {
    setTimeout(function () { dismissToast(toast); }, duration);
  }
}

function dismissToast(toast) {
  if (toast.classList.contains('dismissing')) return;
  toast.classList.add('dismissing');
  setTimeout(function () { toast.remove(); }, 260);
}

// ── Loading Overlay ────────────────────────────────

function showLoading() {
  document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('loadingOverlay').classList.add('hidden');
}

// ── Confirm Dialog ────────────────────────────────

function showConfirm(message) {
  return new Promise(function (resolve) {
    el.confirmMessage.innerHTML = message;
    el.confirmOverlay.classList.remove('hidden');

    function cleanup() {
      el.confirmOverlay.classList.add('hidden');
      el.confirmYes.removeEventListener('click', onYes);
      el.confirmNo.removeEventListener('click', onNo);
      el.confirmOverlay.removeEventListener('click', onOverlay);
    }

    function onYes() {
      cleanup();
      resolve(true);
    }

    function onNo() {
      cleanup();
      resolve(false);
    }

    function onOverlay(e) {
      if (e.target === el.confirmOverlay) {
        cleanup();
        resolve(false);
      }
    }

    el.confirmYes.addEventListener('click', onYes);
    el.confirmNo.addEventListener('click', onNo);
    el.confirmOverlay.addEventListener('click', onOverlay);
  });
}

// ── Helpers ────────────────────────────────────────

function escapeHtml(str) {
  if (typeof str !== 'string' && typeof str !== 'number') return '';
  var div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function tryParseJSON(str, fallback) {
  try { return JSON.parse(str); } catch (_) { return fallback; }
}

function scrollToBottom() {
  requestAnimationFrame(function () {
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
  });
}

function closeMobileSidebar() {
  if (window.innerWidth < 768) {
    el.sidebar.classList.remove('open');
    el.sidebarOverlay.classList.add('hidden');
  }
}

// ── Session Rename ─────────────────────────────────

function startRename(sessionId, titleEl) {
  if (!sessionId) return;

  var currentTitle = state.sessions.filter(function (s) { return s.id === sessionId; })[0];
  currentTitle = currentTitle ? currentTitle.title : '';

  var input = document.createElement('input');
  input.type = 'text';
  input.className = 'rename-input';
  input.value = currentTitle;
  input.maxLength = 120;

  titleEl.textContent = '';
  titleEl.appendChild(input);
  input.focus();
  input.select();

  function commitRename() {
    var val = input.value.trim();
    if (val && val !== currentTitle) {
      renameSession(sessionId, val, titleEl);
    } else {
      cancelRename(titleEl, currentTitle);
    }
  }

  function cancelRename(el, fallback) {
    el.textContent = fallback;
  }

  input.addEventListener('blur', function () {
    commitRename();
  });

  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      input.blur();
    } else if (e.key === 'Escape') {
      cancelRename(titleEl, currentTitle);
    }
  });
}

async function renameSession(sessionId, newTitle, titleEl) {
  try {
    await api.patch('/api/chat/sessions/' + sessionId, { title: newTitle });
    // Update local state
    for (var i = 0; i < state.sessions.length; i++) {
      if (state.sessions[i].id === sessionId) {
        state.sessions[i].title = newTitle;
        break;
      }
    }
    titleEl.textContent = newTitle;
  } catch (err) {
    showToast('Failed to rename session: ' + err.message, 'error');
    titleEl.textContent = newTitle || 'Untitled';
    // Reload to get server state
    loadSessions();
  }
}

// ── Render Functions ───────────────────────────────

function renderSidebar() {
  // ── Sessions ──
  if (state.sessions.length === 0) {
    el.sessionList.innerHTML =
      '<li class="empty-hint">No conversations yet</li>';
  } else {
    el.sessionList.innerHTML = state.sessions.map(function (s) {
      var active = s.id === state.activeSessionId ? ' active' : '';
      return (
        '<li class="session-item' + active + '" data-id="' + s.id + '">' +
          '<span class="session-title" data-rename="1">' + escapeHtml(s.title) + '</span>' +
          '<span class="session-delete" data-action="delete-session" data-id="' + s.id + '" title="Delete conversation">✕</span>' +
        '</li>'
      );
    }).join('');
  }

  // ── Documents ──
  if (state.documents.length === 0) {
    el.documentList.innerHTML = '';
    el.emptyDocs.classList.remove('hidden');
  } else {
    el.emptyDocs.classList.add('hidden');
    el.documentList.innerHTML = state.documents.map(function (d) {
      var title = d.title || '';
      return (
        '<li class="document-item" title="' + escapeHtml(title) + '">' +
          '<span class="doc-name">' + escapeHtml(title) + '</span>' +
          '<span class="doc-delete" data-action="delete-document" data-title="' + escapeHtml(title) + '" title="Delete document">✕</span>' +
        '</li>'
      );
    }).join('');
  }

  // ── Connection status ──
  el.connectionDot.className = 'status-dot ' + (state.apiConnected ? 'connected' : 'disconnected');
  el.connectionText.textContent = state.apiConnected ? 'Connected' : 'Disconnected';
}

function renderMessages() {
  if (state.messages.length === 0) {
    el.messagesList.innerHTML =
      '<div class="empty-hint" style="text-align:center;padding:48px 16px;color:var(--text-muted)">' +
        'No messages yet. Start the conversation.' +
      '</div>';
    return;
  }

  el.messagesList.innerHTML = state.messages.map(function (msg) {
    var isUser = msg.role === 'user';
    var content = escapeHtml(msg.content);

    var sourcesHtml = '';
    if (!isUser && msg.sources) {
      var sources = Array.isArray(msg.sources) ? msg.sources : tryParseJSON(msg.sources, []);
      if (sources.length > 0) {
        var items = sources.map(function (s) {
          var pct = Math.round((s.similarity || 0) * 100);
          return (
            '<div class="source-item">' +
              '<div class="source-similarity">' + pct + '% relevance</div>' +
              '<div>' + escapeHtml(s.content) + '</div>' +
            '</div>'
          );
        }).join('');
        sourcesHtml =
          '<details class="source-toggle">' +
            '<summary>View sources (' + sources.length + ')</summary>' +
            '<div class="source-list">' + items + '</div>' +
          '</details>';
      }
    }

    return (
      '<div class="message ' + (isUser ? 'user' : 'assistant') + '">' +
        '<div class="message-bubble">' +
          '<div class="msg-content">' + content + '</div>' +
          sourcesHtml +
        '</div>' +
      '</div>'
    );
  }).join('');

  scrollToBottom();
}

function showEmptyState() {
  el.emptyState.classList.remove('hidden');
  el.chatView.classList.add('hidden');
}

function showChatView() {
  el.emptyState.classList.add('hidden');
  el.chatView.classList.remove('hidden');
}

function setTypingIndicator(show) {
  el.typingIndicator.classList.toggle('hidden', !show);
  if (show) scrollToBottom();
}

// ── Action Functions ───────────────────────────────

async function checkHealth() {
  try {
    await api.get('/api/health');
    state.apiConnected = true;
  } catch (_) {
    state.apiConnected = false;
    showToast('Could not connect to server', 'error', 5000);
  }
  renderSidebar();
}

async function loadSessions() {
  try {
    var data = await api.get('/api/chat/sessions');
    state.sessions = data.sessions || [];
  } catch (err) {
    showToast('Failed to load sessions: ' + err.message, 'error');
  }
  renderSidebar();
}

async function loadDocuments() {
  try {
    var data = await api.get('/api/documents');
    var seenTitles = new Set();
    state.documents = (data.documents || []).filter(function (document) {
      var title = document.title || '';
      if (!title || seenTitles.has(title)) return false;
      seenTitles.add(title);
      return true;
    });
  } catch (err) {
    showToast('Failed to load documents: ' + err.message, 'error');
  }
  renderSidebar();
}

async function loadMessages(sessionId) {
  try {
    var data = await api.get('/api/chat/sessions/' + sessionId + '/messages');
    state.messages = data.messages || [];
  } catch (err) {
    state.messages = [];
    showToast('Failed to load messages: ' + err.message, 'error');
  }
  renderMessages();
}

async function newChat() {
  try {
    var data = await api.post('/api/chat/sessions', { title: 'New Chat' });
    var session = data.session;
    state.sessions.unshift(session);
    selectSession(session.id);
    closeMobileSidebar();
  } catch (err) {
    showToast('Failed to create session: ' + err.message, 'error');
  }
}

function selectSession(sessionId) {
  state.activeSessionId = sessionId;
  localStorage.setItem('activeSessionId', sessionId);
  showChatView();
  renderSidebar();
  loadMessages(sessionId);
}

async function deleteSession(sessionId, event) {
  event.stopPropagation();

  try {
    await api.delete('/api/chat/sessions/' + sessionId);
    state.sessions = state.sessions.filter(function (s) { return s.id !== sessionId; });

    if (state.activeSessionId === sessionId) {
      if (state.sessions.length > 0) {
        selectSession(state.sessions[0].id);
      } else {
        state.activeSessionId = null;
        localStorage.removeItem('activeSessionId');
        showEmptyState();
        renderSidebar();
      }
    } else {
      renderSidebar();
    }
  } catch (err) {
    showToast('Failed to delete session: ' + err.message, 'error');
  }
}

async function sendMessage() {
  if (state.isTyping || !state.activeSessionId) return;

  var text = el.messageInput.value.trim();
  if (!text) return;

  // Clear input and reset height
  el.messageInput.value = '';
  el.messageInput.style.height = 'auto';
  el.sendBtn.disabled = true;

  // Optimistically add user message
  state.messages.push({ role: 'user', content: text, sources: null });
  renderMessages();

  // Show typing indicator
  state.isTyping = true;
  setTypingIndicator(true);

  try {
    var data = await api.post('/api/chat/sessions/' + state.activeSessionId + '/messages', {
      query: text,
    });

    // Add assistant response
    state.messages.push({
      role: 'assistant',
      content: data.content,
      sources: data.sources || [],
    });
    renderMessages();

    // Refresh sessions to pick up auto-title
    await loadSessions();
  } catch (err) {
    showToast('Failed to send message: ' + err.message, 'error');
    state.messages.push({
      role: 'assistant',
      content: 'Sorry, something went wrong while processing your message.\n\n**Error:** ' + err.message,
      sources: [],
    });
    renderMessages();
  } finally {
    state.isTyping = false;
    setTypingIndicator(false);
    el.sendBtn.disabled = false;
    el.messageInput.focus();
  }
}

async function uploadDocument(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Only PDF files are supported', 'error');
    return;
  }

  var formData = new FormData();
  formData.append('file', file);

  showLoading();

  try {
    var data = await api.upload('/api/upload', formData);
    showToast('"' + file.name + '" uploaded (' + data.chunks_created + ' chunks)', 'success');
    await loadDocuments();
  } catch (err) {
    showToast('Upload failed: ' + err.message, 'error');
  } finally {
    hideLoading();
  }

  el.fileInput.value = '';
  if (el.fileInput2) el.fileInput2.value = '';
}

async function deleteDocument(title, event) {
  if (event) event.stopPropagation();

  var confirmed = await showConfirm('Are you sure you want to remove <strong>' + escapeHtml(title) + '</strong>?');
  if (!confirmed) return;

  try {
    var data = await api.delete('/api/documents/' + encodeURIComponent(title));
    showToast('"' + title + '" deleted (' + data.chunks_deleted + ' chunks removed)', 'success');
    await loadDocuments();
  } catch (err) {
    showToast('Delete failed: ' + err.message, 'error');
  }
}

// ── Event Binding ──────────────────────────────────

function initEvents() {
  // New Chat buttons
  el.newChatBtn.addEventListener('click', newChat);
  el.emptyNewChatBtn.addEventListener('click', newChat);

  // Send message
  el.sendBtn.addEventListener('click', sendMessage);

  // Enter to send, Shift+Enter for newline
  el.messageInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea + enable/disable send button
  el.messageInput.addEventListener('input', function () {
    el.messageInput.style.height = 'auto';
    el.messageInput.style.height = Math.min(el.messageInput.scrollHeight, 200) + 'px';
    el.sendBtn.disabled = !el.messageInput.value.trim() || state.isTyping;
  });

  // Session list click & dblclick delegation
  el.sessionList.addEventListener('click', function (e) {
    var item = e.target.closest('.session-item');
    if (!item) return;

    var sessionId = Number(item.dataset.id);

    if (e.target.dataset.action === 'delete-session') {
      deleteSession(sessionId, e);
    } else if (e.target.classList.contains('session-title')) {
      selectSession(sessionId);
      closeMobileSidebar();
    }
  });

  // Double-click to rename
  el.sessionList.addEventListener('dblclick', function (e) {
    var titleEl = e.target.closest('.session-title');
    if (!titleEl) return;

    var item = titleEl.closest('.session-item');
    if (!item) return;

    var sessionId = Number(item.dataset.id);
    startRename(sessionId, titleEl);
  });

  // Document list click delegation
  el.documentList.addEventListener('click', function (e) {
    if (e.target.dataset.action === 'delete-document') {
      deleteDocument(e.target.dataset.title, e);
    }
  });

  // File upload (sidebar)
  el.fileInput.addEventListener('change', function () {
    var file = el.fileInput.files[0];
    if (file) uploadDocument(file);
  });

  // File upload (empty state)
  if (el.fileInput2) {
    el.fileInput2.addEventListener('change', function () {
      var file = el.fileInput2.files[0];
      if (file) uploadDocument(file);
    });
  }

  // Mobile menu toggle
  el.menuToggle.addEventListener('click', function () {
    el.sidebar.classList.toggle('open');
    el.sidebarOverlay.classList.toggle('hidden');
  });

  // Overlay click closes sidebar
  el.sidebarOverlay.addEventListener('click', function () {
    el.sidebar.classList.remove('open');
    el.sidebarOverlay.classList.add('hidden');
  });

  // Resize: auto-close mobile sidebar
  window.addEventListener('resize', function () {
    if (window.innerWidth >= 768) {
      el.sidebar.classList.remove('open');
      el.sidebarOverlay.classList.add('hidden');
    }
  });
}

// ── Initialization ─────────────────────────────────

async function init() {
  initEvents();

  // Check health and set connection status
  await checkHealth();

  // Load initial data in parallel
  await Promise.all([loadSessions(), loadDocuments()]);

  // Restore active session from localStorage
  var savedId = localStorage.getItem('activeSessionId');
  if (savedId && state.sessions.some(function (s) { return s.id === Number(savedId); })) {
    selectSession(Number(savedId));
  } else if (state.sessions.length > 0) {
    selectSession(state.sessions[0].id);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
