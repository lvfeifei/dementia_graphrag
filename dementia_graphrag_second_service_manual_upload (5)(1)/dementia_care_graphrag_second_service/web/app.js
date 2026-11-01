const chatThread = document.getElementById('chatThread');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');

let sessionId = createSessionId();
let loading = false;

function createSessionId() {
  const randomPart = window.crypto && crypto.randomUUID
    ? crypto.randomUUID().replaceAll('-', '').slice(0, 16)
    : Math.random().toString(16).slice(2, 18);
  return `web_${Date.now().toString(36)}_${randomPart}`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function safeUrl(value) {
  const raw = String(value || '').trim();
  if (/^(https?:|mailto:|\/|#)/i.test(raw)) {
    return raw;
  }
  return '#';
}

const MEDIA_HOST = 'https://advice-1315324619.cos.ap-beijing.myqcloud.com/';

function normalizeMediaMarkdown(text) {
  let result = String(text || '');

  result = result.replace(
    /<video\b[^>]*\bsrc="(https:\/\/advice-1315324619\.cos\.ap-beijing\.myqcloud\.com\/[^"]+\.mp4)"[^>]*>\s*<\/video>/gi,
    '![视频素材]($1)'
  );

  result = result.replace(
    /(^|[\s(：:])(https:\/\/advice-1315324619\.cos\.ap-beijing\.myqcloud\.com\/[^)\s]+\.(?:jpg|jpeg|png|webp))/gim,
    '$1![图片素材]($2)'
  );

  return result;
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(normalizeMediaMarkdown(text));

  html = html.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, label, url) => {
    const safe = safeUrl(url);
    if (!safe.startsWith(MEDIA_HOST)) {
      return match;
    }
    if (/\.mp4(?:\?|$)/i.test(safe)) {
      return `<video class="answer-video" controls preload="metadata" src="${safe}"></video>`;
    }
    return `<img class="answer-image" src="${safe}" alt="${label}">`;
  });
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
    const safe = safeUrl(url);
    return `<a href="${safe}"${String(safe).startsWith('http') ? ' target="_blank" rel="noopener noreferrer"' : ''}>${label}</a>`;
  });

  return html;
}

function isTableSeparator(line) {
  const cells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
  return cells.length > 0 && cells.every((cell) => /^\s*:?-{2,}:?\s*$/.test(cell));
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function renderTable(lines, index) {
  const headers = splitTableRow(lines[index]);
  const rows = [];
  let cursor = index + 2;

  while (cursor < lines.length && lines[cursor].trim().startsWith('|')) {
    rows.push(splitTableRow(lines[cursor]));
    cursor += 1;
  }

  const head = headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join('');
  const body = rows.map((row) => {
    const cells = headers.map((_, cellIndex) => `<td>${renderInlineMarkdown(row[cellIndex] || '')}</td>`).join('');
    return `<tr>${cells}</tr>`;
  }).join('');

  return { html: `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`, next: cursor };
}

function renderMarkdown(markdown) {
  const lines = String(markdown || '').split(/\r?\n/);
  const htmlParts = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith('```')) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index]);
        index += 1;
      }
      index += 1;
      htmlParts.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      continue;
    }

    if (trimmed.startsWith('|') && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      const table = renderTable(lines, index);
      htmlParts.push(table.html);
      index = table.next;
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      htmlParts.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^(---|\*\*\*|___)\s*$/.test(trimmed)) {
      htmlParts.push('<hr>');
      index += 1;
      continue;
    }

    if (trimmed.startsWith('>')) {
      const quoteLines = [];
      while (index < lines.length && lines[index].trim().startsWith('>')) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ''));
        index += 1;
      }
      htmlParts.push(`<blockquote>${renderInlineMarkdown(quoteLines.join(' '))}</blockquote>`);
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      const items = [];
      while (index < lines.length && /^[-*+]\s+/.test(lines[index].trim())) {
        items.push(`<li>${renderInlineMarkdown(lines[index].trim().replace(/^[-*+]\s+/, ''))}</li>`);
        index += 1;
      }
      htmlParts.push(`<ul>${items.join('')}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(`<li>${renderInlineMarkdown(lines[index].trim().replace(/^\d+\.\s+/, ''))}</li>`);
        index += 1;
      }
      htmlParts.push(`<ol>${items.join('')}</ol>`);
      continue;
    }

    const paragraphLines = [];
    while (
      index < lines.length
      && lines[index].trim()
      && !lines[index].trim().startsWith('#')
      && !lines[index].trim().startsWith('```')
      && !lines[index].trim().startsWith('>')
      && !/^[-*+]\s+/.test(lines[index].trim())
      && !/^\d+\.\s+/.test(lines[index].trim())
      && !lines[index].trim().startsWith('|')
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    htmlParts.push(`<p>${renderInlineMarkdown(paragraphLines.join('<br>'))}</p>`);
  }

  return htmlParts.join('');
}

function addMessage(role, content = '') {
  const message = document.createElement('section');
  message.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '我' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const contentWrap = document.createElement('div');
  contentWrap.className = 'content';

  if (role === 'assistant') {
    contentWrap.classList.add('markdown-body');
    contentWrap.innerHTML = renderMarkdown(content);
  } else {
    contentWrap.textContent = content;
  }

  bubble.appendChild(contentWrap);
  message.appendChild(avatar);
  message.appendChild(bubble);
  chatThread.appendChild(message);
  scrollToBottom();

  return { message, bubble, contentWrap };
}

function addLoadingMessage() {
  const result = addMessage('assistant', '');
  result.contentWrap.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  return result;
}

function renderMeta(container, data) {
  const meta = document.createElement('div');
  meta.className = 'meta-card';

  const problemIds = data.selected_problem_ids || [];
  if (problemIds.length) {
    const title = document.createElement('div');
    title.className = 'meta-title';
    title.textContent = '匹配到的照护问题';
    const chips = document.createElement('div');
    chips.className = 'chip-row';
    chips.innerHTML = problemIds.map((id) => `<span class="chip">${escapeHtml(id)}</span>`).join('');
    meta.appendChild(title);
    meta.appendChild(chips);
  }

  const sources = data.evidence_sources || [];
  if (sources.length) {
    const title = document.createElement('div');
    title.className = 'meta-title';
    title.textContent = '证据来源';
    const list = document.createElement('ul');
    list.className = 'evidence-list';

    list.innerHTML = sources.slice(0, 8).map((source) => {
      const parts = [];
      ['type', 'label', 'title', 'name', 'id'].forEach((key) => {
        if (source[key]) parts.push(`${key}: ${source[key]}`);
      });
      return `<li>${escapeHtml(parts.join(' / ') || JSON.stringify(source))}</li>`;
    }).join('');

    meta.appendChild(title);
    meta.appendChild(list);
  }

  if (meta.children.length) {
    container.appendChild(meta);
  }
}

function extractErrorMessage(error) {
  const detail = error && error.detail;
  if (detail && typeof detail === 'object') {
    return detail.error || detail.hint || JSON.stringify(detail);
  }
  if (detail) {
    return detail;
  }
  return error?.message || '服务暂时不可用，请稍后重试。';
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || loading) return;

  loading = true;
  sendBtn.disabled = true;
  addMessage('user', question);
  questionInput.value = '';
  autoResizeTextarea();

  const loadingMessage = addLoadingMessage();

  try {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 120000);

    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        session_id: sessionId,
        question,
        reset_session: false,
        memory_context_mode: 'auto',
        include_active_problem_ids: true,
        temperature: 0.2,
        max_tokens: 3200
      })
    });

    clearTimeout(timeoutId);
    const data = await response.json();

    loadingMessage.message.remove();

    if (!response.ok || !data.success) {
      const failed = addMessage('error', extractErrorMessage(data));
      failed.contentWrap.textContent = extractErrorMessage(data);
      return;
    }

    const result = addMessage('assistant', data.answer_md || '服务没有返回有效内容。');
    renderMeta(result.bubble, data);
  } catch (error) {
    loadingMessage.message.remove();
    const text = error.name === 'AbortError'
      ? '请求超时。GraphRAG 首次加载模型可能较慢，可稍后重试或查看服务日志。'
      : `无法连接 GraphRAG 服务：${error.message}`;
    const failed = addMessage('error', text);
    failed.contentWrap.textContent = text;
  } finally {
    loading = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

function scrollToBottom() {
  window.requestAnimationFrame(() => {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
  });
}

function autoResizeTextarea() {
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
}

sendBtn.addEventListener('click', sendQuestion);

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendQuestion();
  }
});

questionInput.addEventListener('input', autoResizeTextarea);

resetBtn.addEventListener('click', () => {
  sessionId = createSessionId();
  chatThread.innerHTML = '';
  addMessage('assistant', '已开始新会话。请描述老人最近的照护问题。');
  questionInput.focus();
});

questionInput.focus();
