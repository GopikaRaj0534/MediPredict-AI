/* ==================================================================
   main.js
   Shared front-end behaviour for every page:
     1. Scroll-triggered reveal animations (.reveal elements)
     2. Vital-sign scroll progress bar in the top nav
     3. Mobile nav toggle
   ================================================================== */

document.addEventListener("DOMContentLoaded", () => {

  /* ---------- 1. Scroll reveal ---------- */
  const revealEls = document.querySelectorAll(".reveal");

  if ("IntersectionObserver" in window && revealEls.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -60px 0px" }
    );
    revealEls.forEach((el) => observer.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  }

  /* ---------- 2. Scroll progress "pulse" bar ---------- */
  const pulseFill = document.querySelector(".pulse-progress__fill");

  function updatePulseProgress() {
    if (!pulseFill) return;
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? scrollTop / docHeight : 0;
    pulseFill.style.strokeDashoffset = String(1 - progress);
  }

  window.addEventListener("scroll", updatePulseProgress, { passive: true });
  updatePulseProgress();

  /* ---------- 3. Mobile nav toggle ---------- */
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.querySelector(".site-nav__links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("is-open");
    });
  }

});
