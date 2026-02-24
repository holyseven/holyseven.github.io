const ALLOWED_ORIGINS = [
  'https://holyseven.github.io',
  'http://localhost:4000',
  'http://127.0.0.1:4000',
];

const RATE_LIMIT_WINDOW = 60 * 1000; // 1 minute
const RATE_LIMIT_MAX = 10; // max 10 requests per window per IP
const MAX_NICKNAME_LENGTH = 50;
const MAX_CONTENT_LENGTH = 2000;

// In-memory rate limit store (resets on worker restart, good enough for basic protection)
const rateLimitMap = new Map();

function getCorsHeaders(request) {
  const origin = request.headers.get('Origin') || '';
  if (ALLOWED_ORIGINS.includes(origin)) {
    return {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400',
    };
  }
  return {};
}

function jsonResponse(data, status = 200, corsHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
    },
  });
}

function checkRateLimit(ip) {
  const now = Date.now();
  const entry = rateLimitMap.get(ip);

  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW) {
    rateLimitMap.set(ip, { windowStart: now, count: 1 });
    return true;
  }

  if (entry.count >= RATE_LIMIT_MAX) {
    return false;
  }

  entry.count++;
  return true;
}

async function handleGetComments(request, env) {
  const url = new URL(request.url);
  const slug = url.searchParams.get('slug');
  const corsHeaders = getCorsHeaders(request);

  if (!slug) {
    return jsonResponse({ error: 'Missing slug parameter' }, 400, corsHeaders);
  }

  const { results } = await env.DB.prepare(
    'SELECT id, page_slug, parent_id, nickname, content, is_author, created_at FROM comments WHERE page_slug = ? ORDER BY created_at ASC'
  )
    .bind(slug)
    .all();

  return jsonResponse({ comments: results }, 200, corsHeaders);
}

async function handlePostComment(request, env) {
  const corsHeaders = getCorsHeaders(request);
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';

  if (!checkRateLimit(ip)) {
    return jsonResponse({ error: '请求过于频繁，请稍后再试' }, 429, corsHeaders);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: 'Invalid JSON' }, 400, corsHeaders);
  }

  const { slug, nickname, content, parent_id, author_key } = body;

  if (!slug || !nickname || !content) {
    return jsonResponse({ error: '缺少必填字段' }, 400, corsHeaders);
  }

  if (nickname.length > MAX_NICKNAME_LENGTH) {
    return jsonResponse({ error: `昵称不能超过 ${MAX_NICKNAME_LENGTH} 个字符` }, 400, corsHeaders);
  }

  if (content.length > MAX_CONTENT_LENGTH) {
    return jsonResponse({ error: `评论内容不能超过 ${MAX_CONTENT_LENGTH} 个字符` }, 400, corsHeaders);
  }

  // Validate parent_id exists if provided
  if (parent_id != null) {
    const parent = await env.DB.prepare('SELECT id FROM comments WHERE id = ?').bind(parent_id).first();
    if (!parent) {
      return jsonResponse({ error: '回复的评论不存在' }, 400, corsHeaders);
    }
  }

  const isAuthor = author_key && env.AUTHOR_KEY && author_key === env.AUTHOR_KEY ? 1 : 0;
  const createdAt = new Date().toISOString();

  const result = await env.DB.prepare(
    'INSERT INTO comments (page_slug, parent_id, nickname, content, is_author, created_at) VALUES (?, ?, ?, ?, ?, ?)'
  )
    .bind(slug, parent_id || null, nickname.trim(), content.trim(), isAuthor, createdAt)
    .run();

  const newComment = {
    id: result.meta.last_row_id,
    page_slug: slug,
    parent_id: parent_id || null,
    nickname: nickname.trim(),
    content: content.trim(),
    is_author: isAuthor,
    created_at: createdAt,
  };

  return jsonResponse({ comment: newComment }, 201, corsHeaders);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const corsHeaders = getCorsHeaders(request);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    // Route
    if (url.pathname === '/comments') {
      if (request.method === 'GET') {
        return handleGetComments(request, env);
      }
      if (request.method === 'POST') {
        return handlePostComment(request, env);
      }
    }

    return jsonResponse({ error: 'Not found' }, 404, corsHeaders);
  },
};
