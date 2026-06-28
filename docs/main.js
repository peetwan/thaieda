(function () {
  "use strict";

  // ---- Theme toggle (persisted) ----
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem("thaieda-theme"); } catch (e) {}
  if (stored === "light" || stored === "dark") {
    root.setAttribute("data-theme", stored);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    root.setAttribute("data-theme", "light");
  }

  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("thaieda-theme", next); } catch (e) {}
    });
  }

  // ---- Quickstart tabs ----
  var tabs = document.querySelectorAll(".qs__tab");
  var panels = document.querySelectorAll(".code-block[data-panel]");
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var name = tab.getAttribute("data-tab");
      tabs.forEach(function (t) { t.classList.toggle("is-active", t === tab); });
      panels.forEach(function (p) {
        p.classList.toggle("is-active", p.getAttribute("data-panel") === name);
      });
    });
  });

  // ---- Copy buttons ----
  var copyTargets = {
    cli: "pip install thaieda\nthaieda data.csv -o report.html\nthaieda data.csv --target clicked\nthaieda data.csv --explore\nthaieda data.csv --columns",
    py: 'import thaieda\n\nresult = thaieda.run("data.csv", target_column="clicked")\nprint(result.quality_score)\nresult.to_html("report.html")',
    cta: "pip install thaieda"
  };
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var key = btn.getAttribute("data-copy");
      var text = copyTargets[key] || "";
      var done = function () {
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  });

  // ---- Footer year ----
  var y = document.getElementById("year");
  if (y) { y.textContent = new Date().getFullYear(); }
})();
