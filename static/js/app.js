/* ===================================================================
   SafeDrive AI — dashboard interactions
   =================================================================== */
const API = '';
const COLORS = { Low: '#22c55e', Medium: '#f59e0b', High: '#ef4444' };
const CHART_DEFAULTS = {
  color: '#e8eef9',
  font: { family: 'Inter, sans-serif', size: 11 },
};

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n || 0).toLocaleString();

/* ---------------- Range label syncing ---------------- */
const rangeLabels = {
  hour: 'hourVal',
  visibility_km: 'visibilityVal',
  traffic_density: 'trafficVal',
  avg_speed_kmph: 'speedVal',
  speed_limit_kmph: 'limitVal',
  accident_history_5yr: 'historyVal',
};
Object.entries(rangeLabels).forEach(([inputId, labelId]) => {
  const input = $(inputId);
  if (!input) return;
  const update = () => { $(labelId).textContent = input.value; };
  input.addEventListener('input', update);
  update();
});

/* ---------------- Quick scenarios ---------------- */
const scenarios = {
  clear: {
    hour: 11, visibility_km: 9.5, weather: 'Clear', road_condition: 'Dry',
    road_type: 'Urban', lighting: 'Daylight', traffic_density: 35,
    avg_speed_kmph: 48, speed_limit_kmph: 60, accident_history_5yr: 2,
  },
  monsoon: {
    hour: 21, visibility_km: 3.2, weather: 'Rain', road_condition: 'Wet',
    road_type: 'Highway', lighting: 'Dark', traffic_density: 72,
    avg_speed_kmph: 82, speed_limit_kmph: 90, accident_history_5yr: 14,
  },
  fog: {
    hour: 5, visibility_km: 0.8, weather: 'Fog', road_condition: 'Wet',
    road_type: 'Highway', lighting: 'Dark', traffic_density: 45,
    avg_speed_kmph: 78, speed_limit_kmph: 100, accident_history_5yr: 19,
  },
};
function applyScenario(name) {
  Object.entries(scenarios[name]).forEach(([key, value]) => {
    const el = $(key);
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event('input'));
  });
  document.querySelectorAll('.chip-btn').forEach((b) => b.classList.toggle('active', b.dataset.scenario === name));
}
document.querySelectorAll('.chip-btn').forEach((btn) => {
  btn.addEventListener('click', () => applyScenario(btn.dataset.scenario));
});

/* ---------------- Bootstrap ---------------- */
async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

async function loadStatusAndStats() {
  try {
    const [health, info, summary] = await Promise.all([
      getJSON('/api/health'), getJSON('/api/model/info'), getJSON('/api/data/summary'),
    ]);
    const nav = $('navStatus');
    nav.classList.add('online');
    nav.innerHTML = `<span class="status-dot"></span> Model online · ${info.best_model || 'N/A'}`;

    $('statRecords').textContent = fmt(summary.total_records);
    $('statLocations').textContent = fmt(summary.total_locations);
    const comparison = info.model_comparison || {};
    const best = comparison[info.best_model] || {};
    $('statAccuracy').textContent = best.accuracy ? `${(best.accuracy * 100).toFixed(1)}%` : '—';
    $('statAuc').textContent = best.roc_auc ? best.roc_auc.toFixed(3) : '—';
    $('bestModelPill').textContent = `Best model: ${info.best_model || 'N/A'} · F1 ${best.f1 || '—'}`;

    buildCharts(info, summary);
  } catch (err) {
    console.error(err);
    $('navStatus').textContent = 'Offline — run the Flask server';
  }
}

async function loadLocations() {
  try {
    const locations = await getJSON('/api/locations/high-risk');
    const select = $('location');
    locations.forEach((loc) => {
      const option = document.createElement('option');
      option.value = loc.location;
      option.textContent = `${loc.location} · ${loc.risk_pct}% risk`;
      select.appendChild(option);
    });
    buildMap(locations);
    buildRiskList(locations);
  } catch (err) {
    console.error(err);
  }
}

/* ---------------- Prediction ---------------- */
$('predictForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const btn = $('predictBtn');
  const payload = Object.fromEntries(new FormData(event.target).entries());
  payload.vehicle_count = Number(payload.vehicle_count) || 300;
  btn.disabled = true;
  btn.textContent = 'Analysing…';
  try {
    const res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).error || 'Prediction failed');
    renderPrediction(await res.json());
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Predict Accident Risk';
  }
});

function renderPrediction(data) {
  $('emptyResult').classList.add('hidden');
  $('resultContent').classList.remove('hidden');

  const risk = data.risk_level;
  const badge = $('riskBadge');
  badge.textContent = risk.toUpperCase() + ' RISK';
  badge.className = 'risk-badge ' + risk.toLowerCase();

  const score = Math.max(0, Math.min(100, data.risk_score));
  $('riskScore').textContent = Math.round(score);
  const ring = $('ringFg');
  ring.style.stroke = COLORS[risk];
  ring.style.strokeDashoffset = String(327 - (score / 100) * 327);

  ['Low', 'Medium', 'High'].forEach((level) => {
    const pct = (data.probabilities[level] || 0) * 100;
    const lower = level.toLowerCase();
    $(`prob${level}`).style.width = `${pct}%`;
    $(`prob${level}Val`).textContent = `${pct.toFixed(1)}%`;
  });

  $('factorList').innerHTML = data.key_factors.length
    ? data.key_factors.map((f) => `<span class="factor-chip"><b>${escapeHTML(f.factor)}</b> · ${escapeHTML(f.detail)}</span>`).join('')
    : '<span class="muted">No strong risk factors detected in this reading.</span>';

  $('recommendationList').innerHTML = data.recommendations
    .map((r) => `<li>${escapeHTML(r)}</li>`).join('');
}

function escapeHTML(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ---------------- Charts ---------------- */
let mapInstance = null;

function buildCharts(info, summary) {
  Chart.defaults.color = CHART_DEFAULTS.color;
  Chart.defaults.font = CHART_DEFAULTS.font;

  const riskDist = summary.risk_distribution || {};
  new Chart($('riskChart'), {
    type: 'doughnut',
    data: {
      labels: ['Low', 'Medium', 'High'],
      datasets: [{
        data: ['Low', 'Medium', 'High'].map((k) => riskDist[k] || 0),
        backgroundColor: [COLORS.Low, COLORS.Medium, COLORS.High],
        borderColor: '#111a2b', borderWidth: 3,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } },
  });

  const comparison = info.model_comparison || {};
  new Chart($('modelChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(comparison),
      datasets: [{
        label: 'Macro F1', data: Object.values(comparison).map((m) => m.f1),
        backgroundColor: ['#64748b', '#38bdf8', '#818cf8', '#f472b6'], borderRadius: 6,
      }, {
        label: 'Accuracy', data: Object.values(comparison).map((m) => m.accuracy),
        backgroundColor: '#1e3a5f', borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 1, grid: { color: '#1a253a' } }, x: { grid: { display: false } } },
      plugins: { legend: { position: 'bottom' } },
    },
  });

  const importance = (info.feature_importance || []).slice(0, 8);
  new Chart($('importanceChart'), {
    type: 'bar',
    data: {
      labels: importance.map((f) => f.feature.replace(/_/g, ' ')),
      datasets: [{
        label: 'Importance (F1 drop)', data: importance.map((f) => f.importance),
        backgroundColor: '#38bdf8', borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      scales: { x: { grid: { color: '#1a253a' } }, y: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });

  const weatherDist = summary.weather_distribution || {};
  new Chart($('weatherChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(weatherDist),
      datasets: [{
        label: 'Observations', data: Object.values(weatherDist),
        backgroundColor: ['#22c55e', '#94a3b8', '#38bdf8', '#818cf8', '#f472b6'], borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, grid: { color: '#1a253a' } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });
}

/* ---------------- Risk map ---------------- */
function markerColor(pct) { return pct >= 58 ? COLORS.High : pct >= 45 ? COLORS.Medium : COLORS.Low; }

function buildMap(locations) {
  mapInstance = L.map('riskMap', { scrollWheelZoom: false }).setView([17.75, 74.05], 8);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors', maxZoom: 18,
  }).addTo(mapInstance);

  locations.forEach((loc) => {
    const color = markerColor(loc.risk_pct);
    L.circleMarker([loc.latitude, loc.longitude], {
      radius: 7 + Math.min(12, loc.high_count / 10),
      color: color, weight: 2, fillColor: color, fillOpacity: .45,
    }).addTo(mapInstance).bindPopup(
      `<b>${escapeHTML(loc.location)}</b><br>Risk index: ${loc.risk_pct}%<br>` +
      `High-risk readings: ${loc.high_count}<br>Observations: ${fmt(loc.observations)}`
    );
  });
}

function buildRiskList(locations) {
  $('riskList').innerHTML = locations.map((loc, i) => `
    <div class="risk-item" data-index="${i}">
      <div class="risk-rank">${i + 1}</div>
      <div class="risk-item-info">
        <div class="risk-item-name">${escapeHTML(loc.location)}</div>
        <div class="risk-item-meta">${loc.observations} observations · ${loc.high_count} high-risk</div>
      </div>
      <div class="risk-item-score" style="color:${markerColor(loc.risk_pct)}">${loc.risk_pct}%</div>
    </div>`).join('');

  document.querySelectorAll('.risk-item').forEach((item) => {
    item.addEventListener('click', () => {
      const loc = locations[Number(item.dataset.index)];
      if (mapInstance) {
        mapInstance.setView([loc.latitude, loc.longitude], 11, { animate: true });
      }
    });
  });
}

/* ---------------- Go ---------------- */
loadStatusAndStats();
loadLocations();
