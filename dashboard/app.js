(function (M) {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {grid: null, init: null, doyOverride: null, result: null, ensemble: null, centre: null, secondary: null, day: 0, hover: null, busy: false, lastRequest: null, timer: null};
  const mapIds = ['prime', 'branch', 'difference'];
  const format = value => Number(value.toPrecision(5)).toLocaleString(undefined, {maximumFractionDigits: 4});
  function status(text, error = false) {$('status').textContent = text; $('status').classList.toggle('error', error);}
  function stop() {clearInterval(state.timer); state.timer = null; $('play').textContent = '▶'; $('play').setAttribute('aria-label', 'Play forecast');}
  function invalidate() {
    stop(); state.result = null; state.ensemble = null; state.day = 0; state.hover = null;
    $('day').value = 0; $('day').disabled = true; $('play').disabled = true;
    $('sanity-result').textContent = 'Compare an explicit zero array with the base.';
    status('Settings updated. Run both timelines to see a forecast.'); render();
  }
  function inputDay() {
    const doy = Number($('doy').value);
    if (!$('doy').value || !Number.isInteger(doy) || doy < 1 || doy > 366) throw new Error('Day of year must be an integer from 1 to 366.');
    return doy;
  }
  function settings() {
    const doy = inputDay(), steps = Number($('steps').value), radius = Number($('radius').value), magnitude = Number($('magnitude').value);
    if (!$('steps').value || !Number.isInteger(steps) || steps < 1 || steps > 14) throw new Error('Forecast days must be an integer from 1 to 14.');
    if (!$('radius').value || !Number.isFinite(radius) || radius < 1 || radius > 12) throw new Error('Radius must be between 1 and 12 cells.');
    if (!$('magnitude').value || !Number.isFinite(magnitude) || Math.abs(magnitude) > 10000) throw new Error('Magnitude must be finite and between −10000 and 10000.');
    return {doy, steps, radius, magnitude, channel: Number($('shock-channel').value)};
  }
  function previewSettings() {
    return {radius: Math.max(1, Math.min(12, Number($('radius').value) || 4.5)), magnitude: Number($('magnitude').value) || 0, channel: Number($('shock-channel').value)};
  }
  function frames() {
    if (state.result) return state.result;
    const s = previewSettings(), prime = [state.init[2]];
    const perturbation = M.shockField(state.centre, s.radius, s.channel, s.magnitude, state.secondary);
    const branch = [prime[0].map((field, k) => field.map((row, r) => row.map((v, c) => v + perturbation[k][r][c])))];
    return {prime, branch, difference: M.difference(branch, prime)};
  }
  function legend(id, scale, colors, title) {
    const element = $(id); element.replaceChildren();
    const heading = document.createElement('div'); heading.className = 'legend-title'; heading.textContent = title;
    const bar = document.createElement('div'); bar.className = 'legend-bar'; bar.style.background = `linear-gradient(90deg,${colors.join(',')})`;
    const labels = document.createElement('div'); labels.className = 'legend-labels';
    for (const value of [scale[0], (scale[0] + scale[1]) / 2, scale[1]]) {const label = document.createElement('span'); label.textContent = format(value); labels.append(label);}
    element.append(heading, bar, labels);
  }
  function render() {
    if (!state.grid || !state.init) return;
    const view = Number($('view-channel').value), channel = M.config.channels[view], data = frames();
    const scale = M.extent([data.prime, data.branch], view), deltaExtent = M.extent([data.difference], view);
    const limit = Math.max(Math.abs(deltaExtent[0]), Math.abs(deltaExtent[1]), .0001), diffScale = [-limit, limit];
    for (const id of mapIds) {
      const isDiff = id === 'difference', colors = isDiff ? M.config.differenceColors : channel.colors;
      M.drawMap($(id), state.grid, data[id][state.day][view], isDiff ? diffScale : scale, colors, state.centre, previewSettings().radius, state.hover);
      legend(id + '-legend', isDiff ? diffScale : scale, colors, `${isDiff ? 'Δ ' : ''}${channel.id} · ${channel.unit}`);
    }
    $('result-mode').textContent = state.result ? 'LIVE FORECAST · TWO TIMELINES' : 'INITIAL STATE · PREVIEW';
    $('prime-title').textContent = state.result ? 'Timeline Prime' : 'Last input day';
    $('branch-title').textContent = state.result ? 'Branch' : 'Input with shock';
    $('difference-title').textContent = state.result ? 'Difference' : 'Proposed shock';
    $('prime-mode').textContent = state.result ? 'BASE FORECAST' : 'INPUT';
    $('branch-mode').textContent = state.result ? 'PERTURBED FORECAST' : 'PREVIEW';
    $('difference-mode').textContent = state.result ? 'BRANCH − PRIME' : 'PREVIEW';
    $('day-label').textContent = state.result ? `Day ${state.day + 1}` : 'Input day · shock preview';
    $('day-total').textContent = `/ ${$('steps').value || '—'} days`;
    $('timing').textContent = state.result ? `Prime ${format(state.result.primeMs)} ms · Branch ${format(state.result.branchMs)} ms` : 'No forecast yet';
    const {r, c} = state.centre;
    const lat = state.grid.lat[r][c], lon = state.grid.lon[r][c];
    $('centre-label').textContent = `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? 'E' : 'W'} · [${r}, ${c}]`;
    $('magnitude-unit').textContent = M.config.channels[Number($('shock-channel').value)].unit;
    $('chart-channel').textContent = `${channel.name} · ${channel.id}`;
    const metrics = state.result ? M.responseMetrics(data.difference, view) : null;
    $('response-charts').hidden = !metrics;
    if (metrics) {
      M.drawChart($('footprint-chart'), metrics.map(d => d.footprint), state.day, '#72d5c3', '90% energy footprint (cells)');
      M.drawChart($('peak-chart'), metrics.map(d => d.peak), state.day, '#f4bd72', `Peak absolute response (${channel.unit})`);
    }
    $('footprint-stat').textContent = metrics ? `${metrics[state.day].footprint} cells` : '—';
    $('peak-stat').textContent = metrics ? `${format(metrics[state.day].peak)} ${channel.unit}` : '—';
    if (state.hover) {
      const {r, c} = state.hover;
      $('hover').textContent = `${state.grid.lat[r][c].toFixed(2)}°N, ${state.grid.lon[r][c].toFixed(2)}°E · Prime ${format(data.prime[state.day][view][r][c])} · Branch ${format(data.branch[state.day][view][r][c])} · Δ ${format(data.difference[state.day][view][r][c])} ${channel.unit}`;
    } else $('hover').textContent = 'Hover a cell to inspect all three timelines.';
    if (state.impactPanel) state.impactPanel.render();
    renderEnsemble();
  }
  function renderEnsemble() {
    $('ensemble-panel').hidden = !state.result;
    $('run-ensemble').disabled = state.busy || !state.grid.mask;
    $('ensemble-results').hidden = true;
    if (!state.result) return;
    if (!state.grid.mask) {
      $('ensemble-status').textContent = 'Land sensitivity estimates require the land mask.';
      return;
    }
    if (!state.ensemble) return;
    try {
      const {members} = state.ensemble;
      const heatInput = $('impact-heat').value, rainInput = $('impact-heavy').value;
      if (!heatInput || !rainInput) throw new Error('Enter the heat and heavy-rain thresholds.');
      const thresholds = M.impactThresholds({heatC: Number(heatInput), heavyRainMm: Number(rainInput)});
      const key = `${thresholds.heatC}/${thresholds.heavyRainMm}`;
      if (key !== state.ensemble.thresholds) {
        state.ensemble.summary = M.summarizeEnsemble(members, state.grid.mask, thresholds);
        state.ensemble.thresholds = key;
      }
      const summary = state.ensemble.summary;
      const temperature = summary.map(day => day.temperature);
      const rain = summary.map(day => day.rain);
      const {heatProbability, heavyProbability} = summary[state.day];
      $('ensemble-results').hidden = false;
      M.drawChart($('ensemble-temperature'), temperature.map(day => day.median), state.day, '#f4bd72', 'Temperature (°C)', temperature);
      M.drawChart($('ensemble-rain'), rain.map(day => day.median), state.day, '#72d5c3', 'Rainfall (mm/day)', rain);
      $('ensemble-status').textContent = `${members.length} live forecasts · mean land-cell exceedance probabilities by day`;
      $('ensemble-exceedance').textContent = `Day ${state.day + 1} · Heat above ${thresholds.heatC} °C: ${(100 * heatProbability).toFixed(1)}% · Rain at or above ${thresholds.heavyRainMm} mm/day: ${(100 * heavyProbability).toFixed(1)}%`;
    } catch (error) {$('ensemble-status').textContent = error.message;}
  }
  function applyPreset() {
    const preset = M.config.presets[$('preset').value];
    if (!preset) {state.secondary = null; $('preset-info').textContent = 'Custom channel and magnitude · raised-cosine bump'; invalidate(); return;}
    $('shock-channel').value = preset.channel; $('view-channel').value = preset.channel; $('magnitude').value = preset.magnitude;
    state.secondary = preset.secondary || null; state.centre = M.nearest(state.grid, preset.lat, preset.lon);
    $('preset-info').textContent = preset.description; invalidate();
  }
  function setBusy(busy) {
    state.busy = busy;
    for (const id of ['run', 'sanity', 'demo', 'spin-up', 'upload', 'upload-rain', 'doy', 'steps', 'preset', 'shock-channel', 'magnitude', 'radius']) $(id).disabled = busy;
    $('run-ensemble').disabled = busy || !state.grid.mask;
    $('run').textContent = busy ? 'Computing timelines…' : 'Run both timelines ↗';
    if (busy) stop();
  }
  async function health() {
    try {
      const body = await M.api('/health'), name = String(body.model_name || 'unnamed model');
      const ready = body.status === 'ready', stub = name.toLowerCase().includes('stub');
      $('health').textContent = !ready ? `Not ready · ${name}` : stub ? `⚠ STUB MODEL · ${name}` : `● Ready · ${name}`;
      $('health').className = 'badge ' + (!ready ? 'error' : stub ? 'warning' : 'ready');
      $('health').title = stub ? 'The server is using a persistence/spread stub, not a trained forecast model.' : M.config.api;
    } catch (error) {$('health').textContent = '● Server offline'; $('health').className = 'badge error'; $('health').title = error.message;}
  }
  async function simulate(sanity = false) {
    let s; try {s = settings();} catch (error) {status(error.message, true); return;}
    setBusy(true); status(sanity ? 'Checking base against an explicit zero-array shock…' : 'Running Timeline Prime, then the perturbed Branch…');
    const payload = {init_states: state.init, doy_sincos: state.doyOverride || M.seasonal(s.doy), steps: s.steps};
    try {
      state.lastRequest = payload; $('export').disabled = false;
      const prime = await M.api('/simulate', payload);
      const pert = sanity ? Array.from({length: 6}, () => M.matrix(() => 0)) : M.shockField(state.centre, s.radius, s.channel, s.magnitude, state.secondary);
      const branchPayload = {...payload, perturbation: pert}; state.lastRequest = branchPayload;
      const branch = await M.api('/simulate', branchPayload);
      if (sanity) {
        const max = M.maxAbsDiff(prime.rollout, branch.rollout), passed = max === 0;
        $('sanity-result').textContent = `${passed ? 'PASS' : 'FAIL'} · max abs diff = ${max}`;
        status(`${passed ? 'PASS' : 'FAIL'}: zero-shock comparison · max absolute difference ${max}.`, !passed);
      } else {
        state.result = {prime: prime.rollout, branch: branch.rollout, difference: M.difference(branch.rollout, prime.rollout),
          payload, primeMs: Number.isFinite(prime.inference_ms) ? prime.inference_ms : 0,
          branchMs: Number.isFinite(branch.inference_ms) ? branch.inference_ms : 0};
        state.ensemble = null; $('ensemble-status').textContent = 'Run the ensemble to compare nearby scenarios.';
        state.day = 0; $('day').value = 0; $('day').max = s.steps - 1; $('day').disabled = false; $('play').disabled = s.steps < 2;
        status(`Both timelines ready · ${s.steps} daily steps. Move the slider or press play to follow the response.`); render();
      }
    } catch (error) {status(`Simulation failed: ${error.message}`, true); if (sanity) $('sanity-result').textContent = 'FAIL · request did not complete';}
    finally {setBusy(false); health();}
  }
  async function runEnsemble() {
    if (!state.result || !state.grid.mask || state.busy) return;
    const {payload, prime} = state.result;
    const members = [prime], count = M.ensemblePerturbedMembers;
    state.ensemble = null; $('ensemble-results').hidden = true;
    setBusy(true);
    try {
      for (let member = 0; member < count; member++) {
        $('ensemble-status').textContent = `Running perturbation ${member + 1} of ${count}…`;
        const perturbation = M.sensitivityPerturbation(payload.init_states[2]);
        const response = await M.api('/simulate', {...payload, perturbation});
        members.push(response.rollout);
      }
      state.ensemble = {members};
      renderEnsemble();
    } catch (error) {$('ensemble-status').textContent = `Ensemble failed: ${error.message}`;}
    finally {setBusy(false);}
  }
  async function spinUp() {
    let doy;
    try {doy = inputDay();} catch (error) {status(error.message, true); return;}
    setBusy(true);
    status('Running 14 days from the synthetic seed…');
    try {
      const payload = {init_states: M.demoWindow(state.grid), doy_sincos: M.seasonal(doy), steps: 14};
      const response = await M.api('/simulate', payload);
      state.init = response.rollout.slice(-3);
      state.doyOverride = null;
      state.lastRequest = payload; $('export').disabled = false;
      $('doy').value = ((doy - 1 + payload.steps) % 365) + 1;
      $('upload').value = '';
      $('window-info').textContent = 'Model-generated · days 12–14 of synthetic spin-up';
      invalidate();
      status('Model-generated start state ready. Run both timelines to forecast from it.');
    } catch (error) {status(`Spin-up failed: ${error.message}`, true);}
    finally {setBusy(false);}
  }
  async function init() {
    M.config.channels.forEach((channel, index) => {
      for (const id of ['view-channel', 'shock-channel']) {const option = document.createElement('option'); option.value = index; option.textContent = `${channel.name} (${channel.id})`; $(id).append(option);}
    });
    $('api-label').textContent = `API ${M.config.api}`;
    state.grid = await M.loadGrid(); state.init = M.demoWindow(state.grid);
    state.impactPanel = M.createImpactPanel(() => ({result: state.result, grid: state.grid, centre: state.centre, radius: previewSettings().radius}));
    $('grid-info').textContent = state.grid.mask ? state.grid.source : state.grid.source.replace('coastline from land mask', 'no land mask · no coastline');
    if (state.grid.warning) $('grid-info').title = state.grid.warning;
    applyPreset(); status('Ready to experiment. Initial maps preview your input and shock; run both timelines for a forecast.');
    $('preset').addEventListener('change', applyPreset);
    $('view-channel').addEventListener('change', render);
    for (const id of ['doy', 'steps', 'radius']) $(id).addEventListener('input', () => {if (id === 'doy') state.doyOverride = null; invalidate();});
    $('magnitude').addEventListener('input', () => {if (!state.secondary) {$('preset').value = 'custom'; $('preset-info').textContent = 'Custom magnitude · raised-cosine bump';} else $('preset-info').textContent = `u10 ${$('magnitude').value} m/s + tcwv ${Number($('magnitude').value) * 2} kg/m²`; invalidate();});
    $('shock-channel').addEventListener('change', () => {$('preset').value = 'custom'; state.secondary = null; $('preset-info').textContent = 'Custom channel · raised-cosine bump'; invalidate();});
    $('demo').addEventListener('click', () => {state.init = M.demoWindow(state.grid); state.doyOverride = null; $('window-info').textContent = 'Synthetic demo · 3 input days'; $('upload').value = ''; invalidate();});
    $('spin-up').addEventListener('click', spinUp);
    $('run-ensemble').addEventListener('click', runEnsemble);
    for (const id of ['impact-heat', 'impact-heavy']) $(id).addEventListener('input', renderEnsemble);
    $('upload').addEventListener('change', async event => {
      const file = event.target.files[0]; if (!file) return;
      if (file.size > 8 * 1024 * 1024) {status('Window JSON exceeds the 8 MiB limit.', true); return;}
      setBusy(true);
      try {
        const parsed = M.parseWindow(JSON.parse(await file.text()), $('upload-rain').value);
        state.init = parsed.init; state.doyOverride = parsed.doy || null;
        if (parsed.doy) {const angle = Math.atan2(parsed.doy[0], parsed.doy[1]); $('doy').value = Math.max(1, Math.round((angle < 0 ? angle + 2 * Math.PI : angle) * 365 / (2 * Math.PI)));}
        $('window-info').textContent = `${file.name} · 3 input days`; invalidate();
      } catch (error) {status(`Upload rejected: ${error.message}`, true);}
      finally {setBusy(false);}
    });
    $('run').addEventListener('click', () => simulate()); $('sanity').addEventListener('click', () => simulate(true));
    $('day').addEventListener('input', () => {state.day = Number($('day').value); render();});
    $('play').addEventListener('click', () => {
      if (state.timer) {stop(); return;}
      if (!state.result) return;
      $('play').textContent = 'Ⅱ'; $('play').setAttribute('aria-label', 'Pause forecast');
      state.timer = setInterval(() => {state.day = (state.day + 1) % state.result.prime.length; $('day').value = state.day; render();}, 700);
    });
    $('export').addEventListener('click', () => {
      if (!state.lastRequest) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(state.lastRequest)], {type: 'application/json'}));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'monsoon-last-request.json';
      document.body.append(anchor); anchor.click(); anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    });
    mapIds.forEach(id => {
      const canvas = $(id);
      canvas.addEventListener('click', event => {if (state.busy) return; const cell = M.pickCell(canvas, state.grid, event); if (cell) {state.centre = cell; invalidate();}});
      canvas.addEventListener('mousemove', event => {state.hover = M.pickCell(canvas, state.grid, event); render();});
      canvas.addEventListener('mouseleave', () => {state.hover = null; render();});
      canvas.addEventListener('keydown', event => {
        const delta = {ArrowUp: [1, 0], ArrowDown: [-1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]}[event.key];
        if (!delta || state.busy) return; event.preventDefault();
        const {r, c} = state.centre, lat = state.grid.lat[r][c], lon = state.grid.lon[r][c];
        state.centre = M.nearest(state.grid, lat + delta[0], lon + delta[1]); invalidate();
      });
    });
    let resizeTimer; window.addEventListener('resize', () => {clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 100);});
    document.addEventListener('visibilitychange', () => {if (document.hidden) stop();});
    health(); setInterval(health, 30000);
  }
  init().catch(error => status(`Dashboard initialization failed: ${error.message}`, true));
})(window.Monsoon);
