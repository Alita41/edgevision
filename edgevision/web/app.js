const byId = (id) => document.getElementById(id);
const number = (value) => Number(value).toLocaleString();

function setText(id, value) {
  const element = byId(id);
  if (element) element.textContent = String(value);
}

function updateClock() {
  setText('clock', new Date().toLocaleTimeString('en-GB', { hour12: false }));
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function makeEventNode(event) {
  const row = document.createElement('div');
  row.className = 'event-item';

  const icon = document.createElement('span');
  icon.className = 'event-icon';
  icon.textContent = '⌗';

  const content = document.createElement('div');
  content.className = 'event-content';

  const title = document.createElement('div');
  title.className = 'event-title';
  const name = String(event.label ?? 'unknown');
  title.textContent = name.charAt(0).toUpperCase() + name.slice(1);

  const metadata = document.createElement('div');
  metadata.className = 'event-meta';
  const timestamp = document.createElement('span');
  timestamp.textContent = new Date(event.confirmed_at * 1000).toLocaleTimeString();
  const track = document.createElement('span');
  track.textContent = `TRACK #${String(event.track_id).padStart(3, '0')}`;
  const source = document.createElement('span');
  source.textContent = String(event.source ?? '').includes('heuristic')
    ? 'HEURISTIC CANDIDATE'
    : `${Math.round(Number(event.confidence) * 100)}% MODEL CONF.`;
  metadata.append(timestamp, track, source);
  content.append(title, metadata);

  const severity = document.createElement('span');
  severity.className = `severity ${event.severity}`;
  severity.textContent = event.severity;
  row.append(icon, content, severity);
  return row;
}

let previousEventIds = '';
let initialSettingsLoaded = false;

async function updateEvents() {
  const payload = await requestJson('/api/v1/events?limit=15');
  setText('event-pill', `${number(payload.total)} EVENTS`);

  const ids = JSON.stringify(payload.events.map((event) => event.id));
  if (ids === previousEventIds) return;
  previousEventIds = ids;

  const container = byId('event-list');
  container.replaceChildren();
  if (payload.events.length) {
    payload.events.forEach((event) => container.append(makeEventNode(event)));
    return;
  }

  const empty = document.createElement('div');
  empty.className = 'empty-state';
  empty.innerHTML = '<span class="empty-mark">⌁</span><strong>Waiting for verified events</strong><p>Confirmed observations will appear here automatically.</p>';
  container.append(empty);
}

function updateControlLabels() {
  setText('conf-out', `${byId('conf').value}%`);
  setText('hits-out', `${byId('hits').value} FRAMES`);
}

async function updateTelemetry() {
  try {
    const data = await requestJson('/api/v1/telemetry');
    setText('fps', data.throughput_fps.toFixed(1));
    setText('latency', data.processing_ms_p95.toFixed(1));
    setText('tracks', number(data.active_tracks));
    setText('event-count', number(data.events));
    setText('frame-label', `FRAME ${String(data.frames).padStart(6, '0')}`);
    setText('source-tag', data.source.toUpperCase());
    setText('source-detail', data.source);
    setText('captured', number(data.captured_frames));
    setText('dropped', number(data.dropped_frames));
    setText('queue', number(data.queue_depth));
    setText('median', `${data.processing_ms_p50.toFixed(2)} ms`);

    const health = data.error ? 'DEGRADED' : data.stream_ended ? 'SOURCE ENDED' : 'OPERATIONAL';
    setText('health', health);
    setText('health-tag', data.error ? 'DEGRADED' : data.stream_ended ? 'SOURCE ENDED' : 'LOCAL / ACTIVE');

    const demo = data.mode === 'heuristic demo';
    setText('runtime-mode', demo ? 'CANDIDATE DEMO' : 'TFLITE INFERENCE');
    setText('mode-badge', demo ? 'HEURISTIC DEMO' : 'TFLITE MODEL');
    if (!demo) {
      setText('model-warning', 'TFLite model active. Low-confidence or ambiguous classifications are withheld from verified alerts. Model performance depends on your training and calibration.');
    }

    if (!initialSettingsLoaded) {
      byId('conf').value = Math.round(data.config.confidence_threshold * 100);
      byId('hits').value = data.config.min_track_hits;
      updateControlLabels();
      initialSettingsLoaded = true;
    }
    await updateEvents();
  } catch (error) {
    setText('health', 'API UNAVAILABLE');
    setText('health-tag', 'OFFLINE');
    console.warn('EdgeVision telemetry:', error.message);
  }
}

async function applySettings() {
  const button = byId('apply');
  button.disabled = true;
  setText('save-msg', 'Applying…');

  try {
    await requestJson('/api/v1/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        confidence_threshold: Number(byId('conf').value) / 100,
        min_track_hits: Number(byId('hits').value),
      }),
    });
    setText('save-msg', '✓ Saved to live runtime');
  } catch (error) {
    setText('save-msg', `Could not save · ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

const navigation = Array.from(document.querySelectorAll('.nav-stack a'));
navigation.forEach((link) => {
  link.addEventListener('click', () => {
    navigation.forEach((item) => item.classList.remove('nav-active'));
    link.classList.add('nav-active');
  });
});

byId('conf').addEventListener('input', updateControlLabels);
byId('hits').addEventListener('input', updateControlLabels);
byId('apply').addEventListener('click', applySettings);

updateClock();
setInterval(updateClock, 1000);
updateTelemetry();
setInterval(updateTelemetry, 1250);
