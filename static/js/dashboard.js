/* dashboard.js
   Small dashboard-only behaviour. main.js still handles the shared
   nav toggle, scroll reveal, and pulse-progress bar. */

document.addEventListener("DOMContentLoaded", () => {
  const rows = document.querySelectorAll(".history-table tbody tr");

  const routeByType = {
    "Symptom Checker": "/symptom-checker",
    "Diabetes": "/diabetes",
    "Heart Disease": "/heart",
    "Stroke": "/stroke",
  };

  rows.forEach((row) => {
    const typeCell = row.children[1];
    if (!typeCell) return;
    const type = typeCell.textContent.split("·")[0].trim();
    const route = routeByType[type];
    if (route) {
      row.style.cursor = "pointer";
      row.addEventListener("click", () => { window.location.href = route; });
    }
  });
});
