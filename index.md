---
layout: default
title: 首页
---
<!-- 首页：展示最近文章列表和简短介绍 -->
<!-- TODO: 修改下方的欢迎语和自我介绍 -->

<div class="home-intro">
  <h1>你好，欢迎来到我的网站</h1>
  <p>这里是我的个人博客，记录技术笔记与生活随想。</p>
</div>

<h2 class="section-title">最近文章</h2>

<ul class="post-list">
  {% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">
      <div class="post-list-title">{{ post.title }}</div>
      <span class="post-list-meta">{{ post.date | date: "%Y-%m-%d" }}</span>
      {% if post.excerpt %}
      <p class="post-list-excerpt">{{ post.excerpt | strip_html | truncate: 100 }}</p>
      {% endif %}
    </a>
  </li>
  {% endfor %}
</ul>
