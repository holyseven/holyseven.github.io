# 评论系统 — Cloudflare Worker

基于 Cloudflare Workers + D1 的轻量评论后端。

## 部署步骤

### 1. 安装 Wrangler CLI

```bash
npm install -g wrangler
wrangler login
```

### 2. 创建 D1 数据库

```bash
wrangler d1 create comments-db
```

将返回的 `database_id` 填入 `wrangler.toml` 中替换 `YOUR_DATABASE_ID_HERE`。

### 3. 初始化数据库表

```bash
wrangler d1 execute comments-db --file=schema.sql
```

### 4. 设置作者密钥

```bash
wrangler secret put AUTHOR_KEY
```

输入一个只有你知道的密钥字符串，用于在评论时验证作者身份。

### 5. 部署 Worker

```bash
wrangler deploy
```

部署成功后会返回 Worker 的 URL（形如 `https://comments-worker.your-subdomain.workers.dev`）。

### 6. 配置前端

将 Worker URL 填入 Jekyll 项目的 `_config.yml`：

```yaml
comments_api: "https://comments-worker.your-subdomain.workers.dev"
```

## API

### GET /comments?slug=xxx

获取某篇文章的全部评论。

### POST /comments

发表评论。请求体：

```json
{
  "slug": "/blog/my-post/",
  "nickname": "昵称",
  "content": "评论内容",
  "parent_id": null,
  "author_key": ""
}
```

- `parent_id`：可选，回复某条评论时传入其 ID
- `author_key`：可选，传入正确的密钥会标记为作者评论
