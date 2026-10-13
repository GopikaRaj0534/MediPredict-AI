/*
 * theme-toggle.js
 * Applies/persists the dark-mode preference. The initial theme is set by
 * an inline script in <head> (see base.html) so there's no flash of the
 * wrong theme on page load; this file only wires up the toggle button.
 */
document.addEventListener("DOMContentLoaded", function () {
  const btn = document.getElementById("themeToggle");
  if (!btn) return;

  function label(theme) {
    return theme === "dark" ? "☀️ Light mode" : "🌙 Dark mode";
  }

  btn.textContent = label(document.documentElement.getAttribute("data-theme") || "light");

  btn.addEventListener("click", function () {
    const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("mediPredictTheme", next);
    } catch (e) {
      /* localStorage unavailable (private mode etc.) — theme just won't persist */
    }
    btn.textContent = label(next);
  });
});
