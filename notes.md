---
layout: default
title: 笔记
permalink: /notes/
---
<!-- 笔记列表页：展示所有短篇随想 -->

<div class="home-intro">
  <h1>笔记</h1>
  <p>一些零散的想法和短篇随想。</p>
</div>

<ul class="notes-list">
  {% assign sorted_notes = site.notes | sort: "date" | reverse %}
  {% for note in sorted_notes %}
  <li>
    <a href="{{ note.url | relative_url }}">{{ note.title }}</a>
    <span class="notes-list-date">{{ note.date | date: "%Y-%m-%d" }}</span>
  </li>
  {% endfor %}
</ul>
