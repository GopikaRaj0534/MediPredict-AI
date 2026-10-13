/**
 * hospitals.js
 * -------------
 * FEATURE 2 — Nearby Hospitals (frontend).
 *
 * 1. Asks the browser for the user's location (Geolocation API).
 * 2. Calls GET /nearby_hospitals?lat=..&lng=.. which uses the Google
 *    Places API server-side if GOOGLE_PLACES_API_KEY is configured.
 * 3. If that's not configured (or returns nothing / fails), falls back to
 *    a plain "Open Nearby Hospitals in Google Maps" link, as specified.
 */

(function () {
  const section = document.getElementById("hospitalsSection");
  if (!section) return; // Only present on disease_result.html / result.html

  const statusEl = document.getElementById("hospitalsStatus");
  const gridEl = document.getElementById("hospitalsGrid");
  const findBtn = document.getElementById("findHospitalsBtn");
  const fallbackLink = document.getElementById("hospitalsFallbackLink");

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function showFallback(lat, lng) {
    const url = (lat != null && lng != null)
      ? `https://www.google.com/maps/search/hospitals+near+me/@${lat},${lng},14z`
      : "https://www.google.com/maps/search/hospitals+near+me";
    fallbackLink.href = url;
    fallbackLink.hidden = false;
  }

  function renderHospitals(hospitals) {
    gridEl.innerHTML = "";
    hospitals.forEach((h) => {
      const card = document.createElement("div");
      card.className = "hospital-card";

      const distance = h.distance_km != null ? `${h.distance_km} km away` : "";
      const rating = h.rating != null ? `★ ${h.rating}` : "";

      card.innerHTML = `
        <p class="hospital-card__name">${escapeHtml(h.name)}</p>
        ${h.address ? `<p class="hospital-card__meta">${escapeHtml(h.address)}</p>` : ""}
        <p class="hospital-card__meta">${[distance, rating].filter(Boolean).join(" &middot; ")}</p>
        <a class="hospital-card__link" href="${h.maps_url}" target="_blank" rel="noopener">Open in Google Maps</a>
      `;
      gridEl.appendChild(card);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  findBtn.addEventListener("click", () => {
    if (!navigator.geolocation) {
      setStatus("Your browser doesn't support location access.");
      showFallback();
      return;
    }

    setStatus("Requesting your location…");
    findBtn.disabled = true;

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude: lat, longitude: lng } = position.coords;

        try {
          const resp = await fetch(`/nearby_hospitals?lat=${lat}&lng=${lng}`);
          const data = await resp.json();

          if (data.configured && data.hospitals && data.hospitals.length) {
            setStatus(`Found ${data.hospitals.length} hospitals near you.`);
            renderHospitals(data.hospitals);
          } else {
            setStatus("Showing hospitals via Google Maps.");
            showFallback(lat, lng);
          }
        } catch (err) {
          setStatus("Showing hospitals via Google Maps.");
          showFallback(lat, lng);
        } finally {
          findBtn.disabled = false;
        }
      },
      () => {
        setStatus("Location permission denied — you can still search manually.");
        showFallback();
        findBtn.disabled = false;
      },
      { timeout: 10000 }
    );
  });
})();
