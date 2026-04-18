/*
 * 文章页脚本：TOC 生成 + 面板切换
 */
(function () {
  // ---- TOC 生成 ----
  var tocNav = document.getElementById('toc-nav');
  var postContent = document.querySelector('.post-content');

  if (tocNav && postContent) {
    var headings = postContent.querySelectorAll('h2, h3');
    if (headings.length > 0) {
      var ul = document.createElement('ul');
      ul.className = 'toc-list';

      headings.forEach(function (h, i) {
        // 确保标题有 id，方便锚点跳转
        if (!h.id) {
          h.id = 'heading-' + i;
        }

        var li = document.createElement('li');
        li.className = h.tagName === 'H3' ? 'toc-item toc-h3' : 'toc-item';

        var a = document.createElement('a');
        a.className = 'toc-link';
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        a.setAttribute('data-target', h.id);

        li.appendChild(a);
        ul.appendChild(li);
      });

      tocNav.appendChild(ul);

      // ---- 滚动高亮 ----
      var tocLinks = tocNav.querySelectorAll('.toc-link');
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            tocLinks.forEach(function (link) {
              link.classList.remove('active');
            });
            var active = tocNav.querySelector('[data-target="' + entry.target.id + '"]');
            if (active) active.classList.add('active');
          }
        });
      }, {
        rootMargin: '-' + (getComputedStyle(document.documentElement).getPropertyValue('--nav-height').trim() || '56px') + ' 0px -60% 0px',
        threshold: 0
      });

      headings.forEach(function (h) { observer.observe(h); });
    }
  }

  // ---- 面板切换 ----
  var tocToggle = document.getElementById('toc-toggle');
  var tocPanel = document.getElementById('toc-panel');
  var commentsToggle = document.getElementById('comments-toggle');
  var commentsPanel = document.getElementById('comments-panel');

  if (tocToggle && tocPanel) {
    tocToggle.addEventListener('click', function () {
      var opening = !tocPanel.classList.contains('open');
      tocPanel.classList.toggle('open');
      tocToggle.classList.toggle('active');
      // 关闭另一侧
      if (opening && commentsPanel && commentsPanel.classList.contains('open')) {
        commentsPanel.classList.remove('open');
        commentsToggle.classList.remove('active');
      }
    });
  }

  if (commentsToggle && commentsPanel) {
    commentsToggle.addEventListener('click', function () {
      var opening = !commentsPanel.classList.contains('open');
      commentsPanel.classList.toggle('open');
      commentsToggle.classList.toggle('active');
      // 关闭另一侧
      if (opening && tocPanel && tocPanel.classList.contains('open')) {
        tocPanel.classList.remove('open');
        tocToggle.classList.remove('active');
      }
    });
  }
})();
