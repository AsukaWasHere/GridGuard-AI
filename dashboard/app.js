/**
 * GridGuard-AI: Web Dashboard Controller
 * Real-time Chart.js telemetry, dynamic KPI cards, TinyML status badges,
 * alert banners, and interactive bench testing.
 */

// API Configuration (Supports relative path when served by Flask or absolute URL for standalone file)
const API_BASE = window.location.origin.includes("5000") 
  ? window.location.origin 
  : "http://localhost:5000";

// State Management
let telemetryChart = null;
let isPaused = false;
let isAudioMuted = false;
let audioContext = null;
let peakFeederAmps = 0.0;
let lastKnownAlertCount = 0;
let pollingIntervalId = null;

const MAX_CHART_POINTS = 35;

// Audio Alert Synthesizer (Web Audio API - no external mp3 files required)
function playAlertChirp() {
  if (isAudioMuted) return;
  try {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }

    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, audioContext.currentTime); // A5
    osc.frequency.exponentialRampToValueAtTime(440, audioContext.currentTime + 0.25);

    gain.gain.setValueAtTime(0.2, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.25);

    osc.connect(gain);
    gain.connect(audioContext.destination);

    osc.start();
    osc.stop(audioContext.currentTime + 0.28);
  } catch (e) {
    console.warn("Web Audio API not allowed yet by user interaction policy.");
  }
}

// Chart.js Setup
function initChart() {
  const ctx = document.getElementById("telemetryChart").getContext("2d");

  telemetryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Feeder Current (GPIO32)",
          data: [],
          borderColor: "#00f2fe",
          backgroundColor: "rgba(0, 242, 254, 0.08)",
          borderWidth: 2.2,
          fill: true,
          tension: 0.3,
          pointRadius: 2,
          pointHoverRadius: 5,
        },
        {
          label: "Metered Current (GPIO33)",
          data: [],
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.08)",
          borderWidth: 2.2,
          fill: true,
          tension: 0.3,
          pointRadius: 2,
          pointHoverRadius: 5,
        },
        {
          label: "Discrepancy (ΔI)",
          data: [],
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244, 63, 94, 0.15)",
          borderWidth: 2.5,
          borderDash: [5, 4],
          fill: false,
          tension: 0.2,
          pointRadius: 3,
          pointHoverRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 250
      },
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          display: false // Using our custom HTML legend
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          titleColor: "#94a3b8",
          bodyColor: "#f8fafc",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: function(context) {
              return ` ${context.dataset.label}: ${context.parsed.y.toFixed(3)} A (${(context.parsed.y * 1000).toFixed(1)} mA)`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: "rgba(255, 255, 255, 0.04)"
          },
          ticks: {
            color: "#64748b",
            font: { family: "'JetBrains Mono', monospace", size: 10 }
          }
        },
        y: {
          min: 0,
          grid: {
            color: "rgba(255, 255, 255, 0.06)"
          },
          ticks: {
            color: "#94a3b8",
            font: { family: "'JetBrains Mono', monospace", size: 11 },
            callback: function(val) {
              return val.toFixed(2) + " A";
            }
          }
        }
      }
    }
  });
}

// Push New Telemetry Data Point to Chart
function pushChartData(timeLabel, feederA, meterA, diffA) {
  if (isPaused || !telemetryChart) return;

  const labels = telemetryChart.data.labels;
  const feederData = telemetryChart.data.datasets[0].data;
  const meterData = telemetryChart.data.datasets[1].data;
  const diffData = telemetryChart.data.datasets[2].data;

  labels.push(timeLabel);
  feederData.push(feederA);
  meterData.push(meterA);
  diffData.push(diffA);

  if (labels.length > MAX_CHART_POINTS) {
    labels.shift();
    feederData.shift();
    meterData.shift();
    diffData.shift();
  }

  telemetryChart.update("none");
}

// Update UI with latest reading
function updateDashboardUI(data) {
  const reading = data.latest;
  const esp32Status = data.esp32_connection_status;
  const simStatus = data.simulation_status;

  if (!reading) return;

  const totalA = reading.total_current_A || 0.0;
  const meterA = reading.meter_current_A || 0.0;
  const diffA = reading.difference_A || 0.0;
  const totalAdc = reading.total_adc || 3102.0;
  const meterAdc = reading.metered_adc || 1551.0;
  const deltaTot = reading.delta_total || 0.0;
  const deltaMet = reading.delta_metered || 0.0;
  const pred = reading.predicted_status || reading.prediction || "UNKNOWN";
  const conf = reading.confidence;
  const ruleStatus = reading.rule_based_status || "NORMAL";
  const source = reading.data_source || "none";

  // 1. Feeder Current Card & Raw ADC
  document.getElementById("feederCurrentVal").textContent = totalA.toFixed(3);
  document.getElementById("feederMilliVal").textContent = `${(totalA * 1000).toFixed(1)} mA`;
  const feederAdcElem = document.getElementById("feederAdcVal");
  if (feederAdcElem) {
    feederAdcElem.textContent = `${totalAdc.toFixed(0)} ADC (Δ${deltaTot >= 0 ? '+' : ''}${deltaTot.toFixed(1)})`;
  }

  // 2. Metered Current Card & Raw ADC
  document.getElementById("meterCurrentVal").textContent = meterA.toFixed(3);
  document.getElementById("meterMilliVal").textContent = `${(meterA * 1000).toFixed(1)} mA`;
  const meterAdcElem = document.getElementById("meterAdcVal");
  if (meterAdcElem) {
    meterAdcElem.textContent = `${meterAdc.toFixed(0)} ADC (Δ${deltaMet >= 0 ? '+' : ''}${deltaMet.toFixed(1)})`;
  }

  // 3. Current Discrepancy Card
  document.getElementById("diffCurrentVal").textContent = diffA.toFixed(3);
  const diffBadge = document.getElementById("diffBadge");
  const diffCard = document.getElementById("cardDiff");
  const diffAdcElem = document.getElementById("diffAdcVal");
  const diffAdcCounts = deltaTot - deltaMet;
  if (diffAdcElem) {
    diffAdcElem.textContent = `Δ ${diffAdcCounts >= 0 ? '+' : ''}${diffAdcCounts.toFixed(1)} ADC`;
  }

  const isBypass = (pred === "POSSIBLE_THEFT" || pred === "POSSIBLE_BYPASS");

  if (isBypass) {
    diffBadge.textContent = "THEFT ALERT";
    diffBadge.className = "status-indicator-mini theft";
    diffCard.classList.add("alert-active");
  } else {
    diffBadge.textContent = "BALANCED";
    diffBadge.className = "status-indicator-mini";
    diffCard.classList.remove("alert-active");
  }

  // 4. Operating Status & Random Forest Inference Card
  const heroBadge = document.getElementById("statusHeroBadge");
  const heroText = document.getElementById("statusHeroText");
  const heroIcon = document.getElementById("statusHeroIcon");
  const confBar = document.getElementById("confidenceBar");
  const confText = document.getElementById("confidenceText");
  const windowStatusElem = document.getElementById("windowStatusVal");
  const modelVerElem = document.getElementById("modelVerVal");

  if (windowStatusElem) {
    windowStatusElem.textContent = reading.window_status ? `${reading.window_status} Samples` : "5/5 Window";
  }
  if (modelVerElem) {
    modelVerElem.textContent = reading.model_version ? `v${reading.model_version} (RF)` : "Random Forest";
  }

  if (conf !== null && conf !== undefined) {
    confBar.style.width = `${Math.round(conf * 100)}%`;
    confText.textContent = `${(conf * 100).toFixed(1)}%`;
  } else {
    confBar.style.width = "0%";
    confText.textContent = "Warming Up";
  }

  heroBadge.className = "status-badge-big";

  if (pred === "POSSIBLE_THEFT" || pred === "POSSIBLE_BYPASS") {
    heroBadge.classList.add("badge-theft");
    heroText.textContent = "POWER THEFT DETECTED";
    heroIcon.className = "fa-solid fa-triangle-exclamation";
    triggerAlertBanner(diffA, diffAdcCounts);
  } else if (reading.detection_state === "VERIFYING") {
    heroBadge.classList.add("badge-warmup");
    const secs = Math.min(reading.verify_elapsed_s || 0, reading.confirm_seconds || 0).toFixed(0);
    heroText.textContent = `VERIFYING LOAD INCREASE… ${secs}/${(reading.confirm_seconds || 0).toFixed(0)}s`;
    heroIcon.className = "fa-solid fa-hourglass-half";
    hideAlertBanner();
  } else if (pred === "WARMING_UP") {
    heroBadge.classList.add("badge-warmup");
    heroText.textContent = `WARMING UP (${reading.window_status || 'Buffering'})`;
    heroIcon.className = "fa-solid fa-hourglass-half";
    hideAlertBanner();
  } else if (pred === "POWER_LOSS") {
    heroBadge.classList.add("badge-powerloss");
    heroText.textContent = "POWER LOSS";
    heroIcon.className = "fa-solid fa-power-off";
    hideAlertBanner();
  } else if (pred === "NORMAL") {
    heroBadge.classList.add("badge-normal");
    heroText.textContent = "NORMAL";
    heroIcon.className = "fa-solid fa-circle-check";
    hideAlertBanner();
  } else {
    heroBadge.classList.add("badge-powerloss");
    heroText.textContent = pred;
    heroIcon.className = "fa-solid fa-circle-question";
    hideAlertBanner();
  }

  // 5. Navigation Badges (Hardware vs Simulation Source & ESP32 Link)
  const sourceDot = document.getElementById("sourceDot");
  const sourceText = document.getElementById("sourceText");

  if (source === "hardware") {
    sourceDot.className = "status-dot dot-active";
    sourceText.textContent = "ESP32 HARDWARE";
    sourceText.style.color = "#38bdf8";
  } else if (source === "simulation") {
    sourceDot.className = "status-dot dot-inactive";
    sourceText.textContent = "SIMULATOR STREAM";
    sourceText.style.color = "#a78bfa";
  } else {
    sourceDot.className = "status-dot dot-offline";
    sourceText.textContent = "IDLE (NO STREAM)";
    sourceText.style.color = "#94a3b8";
  }

  // ESP32 Link Status
  const esp32Dot = document.getElementById("esp32Dot");
  const esp32Text = document.getElementById("esp32Text");

  if (esp32Status === "CONNECTED") {
    esp32Dot.className = "status-dot dot-active";
    esp32Text.textContent = "CONNECTED (ONLINE)";
    esp32Text.style.color = "#34d399";
  } else if (esp32Status === "STALE") {
    esp32Dot.className = "status-dot dot-inactive";
    esp32Text.textContent = "SIGNAL STALE (>10s)";
    esp32Text.style.color = "#fbbf24";
  } else {
    esp32Dot.className = "status-dot dot-offline";
    esp32Text.textContent = "DISCONNECTED";
    esp32Text.style.color = "#f87171";
  }

  // Heartbeat time
  const d = new Date();
  document.getElementById("lastUpdatedText").textContent = d.toLocaleTimeString();

  // Push to Chart
  const timeLabel = d.toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
  pushChartData(timeLabel, totalA, meterA, diffA);
}

// Trigger Alert Banner
function triggerAlertBanner(diffA, diffAdc) {
  const banner = document.getElementById("theftAlertBanner");
  const glow = document.getElementById("ambientAlertGlow");
  const text = document.getElementById("alertBannerText");

  banner.classList.remove("hidden");
  glow.classList.add("active");
  const adcMsg = (diffAdc !== undefined) ? `(Δ${diffAdc.toFixed(1)} ADC counts)` : "";
  text.textContent = `Feeder current exceeds metered load by ${(diffA * 1000).toFixed(1)} mA ${adcMsg}! Stable unmetered load confirmed.`;

  playAlertChirp();
}

function hideAlertBanner() {
  const banner = document.getElementById("theftAlertBanner");
  const glow = document.getElementById("ambientAlertGlow");
  banner.classList.add("hidden");
  glow.classList.remove("active");
}

// Fetch Alerts Log
async function fetchAlerts() {
  try {
    const res = await fetch(`${API_BASE}/api/alerts`);
    if (!res.ok) return;
    const data = await res.json();
    const alerts = data.alerts || [];

    const container = document.getElementById("alertListContainer");
    const countBadge = document.getElementById("alertCountBadge");
    countBadge.textContent = `${alerts.length} Alerts`;

    if (alerts.length === 0) {
      container.innerHTML = `
        <div class="empty-alerts">
          <i class="fa-solid fa-shield-cat"></i>
          <p>No theft discrepancies or power outages logged yet.</p>
        </div>
      `;
      return;
    }

    // Render alert items in reverse chronological order
    const html = alerts.slice(-15).reverse().map(a => {
      const timeStr = a.timestamp_iso ? new Date(a.timestamp_iso).toLocaleTimeString() : "Just now";
      const isHigh = a.severity === "HIGH";
      const icon = a.type === "POWER_LOSS" ? "fa-power-off" : "fa-triangle-exclamation";
      return `
        <div class="alert-item ${isHigh ? 'severity-high' : 'severity-medium'}">
          <div class="alert-item-icon">
            <i class="fa-solid ${icon}"></i>
          </div>
          <div class="alert-item-info">
            <div class="alert-item-title">${a.message || a.type}</div>
            <div class="alert-item-desc">Feeder: ${a.total_current_A}A · Meter: ${a.meter_current_A}A · Discrepancy: ${a.difference_A}A [${a.data_source}]</div>
          </div>
          <div class="alert-item-time">${timeStr}</div>
        </div>
      `;
    }).join("");

    container.innerHTML = html;
  } catch (e) {
    console.debug("Could not fetch alerts:", e);
  }
}

// Fetch Latest Telemetry from Backend
async function pollTelemetry() {
  try {
    const res = await fetch(`${API_BASE}/api/readings/latest`);
    if (!res.ok) {
      document.getElementById("lastUpdatedText").textContent = "Backend Offline";
      const heroBadge = document.getElementById("statusHeroBadge");
      const heroText = document.getElementById("statusHeroText");
      if (heroBadge && heroText) {
        heroBadge.className = "status-badge-big badge-powerloss";
        heroText.textContent = "BACKEND ERROR";
      }
      return;
    }
    const data = await res.json();
    updateDashboardUI(data);
  } catch (err) {
    document.getElementById("lastUpdatedText").textContent = "Connecting to Backend...";
    document.getElementById("esp32Text").textContent = "BACKEND DISCONNECTED";
    const heroBadge = document.getElementById("statusHeroBadge");
    const heroText = document.getElementById("statusHeroText");
    if (heroBadge && heroText) {
      heroBadge.className = "status-badge-big badge-powerloss";
      heroText.textContent = "DISCONNECTED";
    }
  }
}

// Interactive Scenario Simulation Button Handler
async function injectSimulation(scenario) {
  try {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario })
    });
    if (res.ok) {
      pollTelemetry();
      fetchAlerts();
    }
  } catch (err) {
    console.error("Error triggering simulation:", err);
  }
}

// Set up UI Event Listeners
function setupEventListeners() {
  // Scenario injection buttons
  document.getElementById("simNormalBtn").addEventListener("click", () => injectSimulation("normal"));
  document.getElementById("simBypassBtn").addEventListener("click", () => injectSimulation("bypass"));
  document.getElementById("simPowerLossBtn").addEventListener("click", () => injectSimulation("power_loss"));

  // Dismiss Banner
  document.getElementById("dismissAlertBtn").addEventListener("click", hideAlertBanner);

  // Audio Mute Toggle
  document.getElementById("muteAudioBtn").addEventListener("click", function() {
    isAudioMuted = !isAudioMuted;
    this.innerHTML = isAudioMuted 
      ? '<i class="fa-solid fa-volume-xmark"></i>' 
      : '<i class="fa-solid fa-volume-high"></i>';
  });

  // Threshold Slider
  const slider = document.getElementById("thresholdSlider");
  const sliderVal = document.getElementById("sliderValDisplay");
  const thresholdDisplay = document.getElementById("thresholdDisplay");

  slider.addEventListener("input", function() {
    const ma = this.value;
    const a = (ma / 1000).toFixed(3);
    sliderVal.textContent = `${ma} mA (${a} A)`;
    thresholdDisplay.textContent = `±${ma} mA`;
  });

  slider.addEventListener("change", async function() {
    const a = (this.value / 1000).toFixed(3);
    try {
      await fetch(`${API_BASE}/api/threshold`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold_A: parseFloat(a) })
      });
    } catch (e) {
      console.warn("Could not update threshold:", e);
    }
  });

  // Toggle Pause Chart
  document.getElementById("togglePauseBtn").addEventListener("click", function() {
    isPaused = !isPaused;
    this.innerHTML = isPaused 
      ? '<i class="fa-solid fa-play"></i> Resume' 
      : '<i class="fa-solid fa-pause"></i> Pause';
  });

  // Clear Chart
  document.getElementById("clearChartBtn").addEventListener("click", function() {
    if (telemetryChart) {
      telemetryChart.data.labels = [];
      telemetryChart.data.datasets.forEach(ds => ds.data = []);
      telemetryChart.update();
    }
  });

  // Clear Alerts
  document.getElementById("clearAlertsBtn").addEventListener("click", async function() {
    try {
      await fetch(`${API_BASE}/api/alerts/clear`, { method: "POST" });
      fetchAlerts();
    } catch (e) {
      console.warn("Could not clear alerts:", e);
    }
  });
}

// Initialization on DOM Load
window.addEventListener("DOMContentLoaded", () => {
  initChart();
  setupEventListeners();

  // Initial polls
  pollTelemetry();
  fetchAlerts();

  // Set up recurring polling loop (1000ms)
  pollingIntervalId = setInterval(() => {
    pollTelemetry();
  }, 1000);

  // Poll alerts every 3 seconds
  setInterval(() => {
    fetchAlerts();
  }, 3000);
});
