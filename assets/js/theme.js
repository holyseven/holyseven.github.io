/*
 * 深色模式切换脚本
 * 读取用户偏好并在 localStorage 中保存选择
 */
(function () {
  var toggle = document.getElementById("theme-toggle");
  var storedTheme = localStorage.getItem("theme");

  // 初始化主题：优先使用用户保存的选择，其次跟随系统
  if (storedTheme) {
    document.documentElement.setAttribute("data-theme", storedTheme);
  } else if (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    document.documentElement.setAttribute("data-theme", "dark");
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme");
      var next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
    });
  }
})();
