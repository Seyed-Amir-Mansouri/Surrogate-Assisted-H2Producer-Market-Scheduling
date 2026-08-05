document.querySelectorAll(".preset-btn").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var days = parseInt(btn.dataset.days, 10);
    var start = document.getElementById("start_day");
    var end = document.getElementById("end_day");
    if (!start || !end) return;
    var s = parseInt(start.value, 10) || 1;
    start.value = s;
    end.value = Math.min(364, s + days - 1);
  });
});

var runForm = document.getElementById("run-form");
if (runForm) {
  runForm.addEventListener("submit", function () {
    var overlay = document.getElementById("processing-overlay");
    if (overlay) overlay.classList.add("visible");
  });
}

/**
 * Client-side country-tab switching (no reload -- every selected country's charts
 * are already rendered server-side). Plotly figures inside a display:none panel
 * render at zero size, so force a relayout the first time a tab becomes visible.
 */
function activateTab(code) {
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    b.classList.toggle("active", b.dataset.country === code);
  });
  document.querySelectorAll(".tab-panel").forEach(function (p) {
    p.classList.toggle("active", p.dataset.country === code);
  });
  var panel = document.querySelector('.tab-panel[data-country="' + code + '"]');
  if (panel) {
    panel.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      if (window.Plotly) window.Plotly.Plots.resize(gd);
    });
  }
}

document.querySelectorAll(".tab-btn").forEach(function (btn) {
  btn.addEventListener("click", function () { activateTab(btn.dataset.country); });
});

var firstTab = document.querySelector(".tab-btn");
if (firstTab) activateTab(firstTab.dataset.country);
