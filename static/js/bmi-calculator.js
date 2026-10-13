/*
 * bmi-calculator.js
 * Drop-in BMI helper for any form containing a `.bmi-calc` block.
 * Lets users enter height + weight in metric or imperial units and
 * auto-fills the hidden/readonly BMI input the ML model actually needs.
 */
(function () {
  function categoryFor(bmi) {
    if (bmi < 18.5) return { label: "Underweight", tone: "info" };
    if (bmi < 25) return { label: "Normal weight", tone: "good" };
    if (bmi < 30) return { label: "Overweight", tone: "warn" };
    return { label: "Obese", tone: "high" };
  }

  function setup(calc) {
    const unitBtns = calc.querySelectorAll(".bmi-unit-btn");
    const groups = calc.querySelectorAll("[data-unit-group]");
    const bmiInput = calc.querySelector(".bmi-final-value");
    const resultEl = calc.querySelector(".bmi-calc__result");
    const manualToggle = calc.querySelector(".bmi-manual-toggle");
    const calcBtn = calc.querySelector(".bmi-calc-btn");

    function showUnit(unit) {
      groups.forEach((g) => {
        g.hidden = g.dataset.unitGroup !== unit;
      });
      unitBtns.forEach((b) => b.classList.toggle("active", b.dataset.unit === unit));
    }

    unitBtns.forEach((btn) => {
      btn.addEventListener("click", () => showUnit(btn.dataset.unit));
    });

    function compute() {
      const activeUnit = calc.querySelector(".bmi-unit-btn.active").dataset.unit;
      let bmi = null;

      if (activeUnit === "metric") {
        const cm = parseFloat(calc.querySelector(".bmi-height-cm").value);
        const kg = parseFloat(calc.querySelector(".bmi-weight-kg").value);
        if (cm > 0 && kg > 0) {
          const m = cm / 100;
          bmi = kg / (m * m);
        }
      } else {
        const ft = parseFloat(calc.querySelector(".bmi-height-ft").value) || 0;
        const inch = parseFloat(calc.querySelector(".bmi-height-in").value) || 0;
        const lb = parseFloat(calc.querySelector(".bmi-weight-lb").value);
        const totalIn = ft * 12 + inch;
        if (totalIn > 0 && lb > 0) {
          bmi = (703 * lb) / (totalIn * totalIn);
        }
      }

      if (bmi === null || !isFinite(bmi) || bmi <= 0) {
        resultEl.textContent = "Enter your height and weight to calculate your BMI.";
        resultEl.className = "bmi-calc__result";
        return;
      }

      const rounded = Math.round(bmi * 10) / 10;
      const cat = categoryFor(rounded);
      bmiInput.value = rounded;
      resultEl.textContent = `Your BMI is ${rounded} — ${cat.label}`;
      resultEl.className = "bmi-calc__result bmi-calc__result--" + cat.tone;
    }

    calcBtn.addEventListener("click", compute);

    if (manualToggle) {
      manualToggle.addEventListener("click", (e) => {
        e.preventDefault();
        const isManual = calc.classList.toggle("bmi-calc--manual");
        bmiInput.readOnly = !isManual;
        manualToggle.textContent = isManual
          ? "Use the calculator instead"
          : "I already know my BMI — let me type it in";
        if (isManual) {
          bmiInput.value = "";
          resultEl.textContent = "";
          bmiInput.focus();
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".bmi-calc").forEach(setup);
  });
})();
