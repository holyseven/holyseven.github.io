# 我的网站

基于 Jekyll 搭建的个人网站，部署在 GitHub Pages 上。风格简洁，支持深色模式，适合中文阅读。

## 文件夹结构

```
.
├── _config.yml          # 站点配置（标题、URL、集合等）
├── _layouts/            # 页面布局模板
│   ├── default.html     #   基础模板（所有页面共用）
│   ├── post.html        #   文章/笔记模板
│   └── page.html        #   独立页面模板
├── _includes/           # 可复用的页面片段
│   ├── nav.html         #   导航栏
│   ├── header.html      #   页头
│   └── footer.html      #   页脚
├── _posts/              # 博客文章（正式发布的）
├── _drafts/             # 草稿（不会被发布）
├── _notes/              # 笔记/随想（短篇内容）
├── _pages/              # 独立页面（关于等）
├── assets/
│   ├── css/style.css    # 主样式文件
│   └── js/theme.js      # 深色模式切换脚本
├── index.md             # 首页
├── notes.md             # 笔记列表页
├── Gemfile              # Ruby 依赖
└── README.md            # 本文件
```

## 本地预览

1. 确保已安装 Ruby（建议 2.7+）。

2. 安装依赖：

   ```bash
   gem install bundler
   bundle install
   ```

3. 启动本地服务器：

   ```bash
   bundle exec jekyll serve
   ```

4. 在浏览器中打开 [http://localhost:4000](http://localhost:4000) 即可预览。

## 如何使用

### 发布新文章

1. 在 `_posts/` 文件夹中创建新文件，命名格式为 `YYYY-MM-DD-标题.md`。
2. 在文件开头添加 front matter：

   ```yaml
   ---
   title: "文章标题"
   date: 2025-02-01
   tags: [标签1, 标签2]
   excerpt: "文章摘要，会显示在首页列表中。"
   ---
   ```

3. 在 front matter 下方用 Markdown 写正文。
4. 提交并推送到 GitHub，文章会自动发布。

### 写草稿

1. 在 `_drafts/` 文件夹中创建文件，**不需要日期前缀**，如 `my-draft.md`。
2. front matter 和普通文章一样，但不需要 `date` 字段。
3. 草稿不会出现在正式网站中。要在本地预览草稿：

   ```bash
   bundle exec jekyll serve --drafts
   ```

4. 写好后，将文件移到 `_posts/` 并加上日期前缀即可发布。

### 添加新笔记

1. 在 `_notes/` 文件夹中创建 `.md` 文件，文件名即为 URL 中的标题。
2. front matter 示例：

   ```yaml
   ---
   title: "笔记标题"
   date: 2025-02-01
   ---
   ```

3. 笔记会自动出现在「笔记」页面的列表中。

### 添加新的独立页面

1. 在 `_pages/` 文件夹中创建 `.md` 文件。
2. front matter 示例：

   ```yaml
   ---
   layout: page
   title: "页面标题"
   permalink: /页面路径/
   ---
   ```

3. 如果需要在导航栏中显示，编辑 `_includes/nav.html` 添加链接。

## 自定义

- **站点信息**：编辑 `_config.yml`，修改标题、描述、URL 等（文件中有 TODO 注释提示）。
- **样式**：编辑 `assets/css/style.css`，配色方案在文件顶部的 CSS 变量中定义。
- **导航栏**：编辑 `_includes/nav.html`，增减导航链接。
- **布局**：编辑 `_layouts/` 下的文件来调整页面结构。
