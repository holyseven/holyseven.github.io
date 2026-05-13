---
layout: page
title: 标签
permalink: /tags/
---

<div class="tag-filter" id="tag-filter">
  <button class="tag-btn active" data-tag="all">全部</button>
  <button class="tag-btn" data-tag="llm-base">LLM 基础</button>
  <button class="tag-btn" data-tag="training">训练与对齐</button>
  <button class="tag-btn" data-tag="eval">评测与基准</button>
  <button class="tag-btn" data-tag="agent">Agent</button>
  <button class="tag-btn" data-tag="generation">生成与创意</button>
  <button class="tag-btn" data-tag="code">代码与多语言</button>
  <button class="tag-btn" data-tag="diffusion">扩散模型</button>
  <button class="tag-btn" data-tag="systems">系统与工程</button>
  <button class="tag-btn" data-tag="bio">生物信息</button>
</div>

{% assign sorted_tags = site.tags | sort %}
{% for tag in sorted_tags %}
<h3 id="tag-{{ tag[0] }}">{{ tag[0] }} <small>({{ tag[1].size }})</small></h3>
<ul class="tag-post-list">
  {% for post in tag[1] %}
  <li data-tags="{{ post.tags | join: ',' }}"><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <span class="post-list-meta">{{ post.date | date: "%Y-%m-%d" }}</span></li>
  {% endfor %}
</ul>
{% endfor %}

<script src="{{ '/assets/js/tags.js' | relative_url }}"></script>
