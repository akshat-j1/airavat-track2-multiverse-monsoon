(function (M) {
  'use strict';
  function responseMetrics(difference, channel) {
    return difference.map(day => {
      const values = day[channel].flat(), energy = values.map(v => v * v).sort((a, b) => b - a);
      const total = energy.reduce((a, b) => a + b, 0);
      let footprint = 0, cumulative = 0;
      if (total > 0) for (const value of energy) {cumulative += value; footprint++; if (cumulative >= .9 * total) break;}
      return {footprint, peak: values.reduce((max, v) => Math.max(max, Math.abs(v)), 0)};
    });
  }
  function maxAbsDiff(a, b) {
    let max = 0;
    for (let d = 0; d < a.length; d++) for (let k = 0; k < 6; k++) for (let r = 0; r < 32; r++) for (let c = 0; c < 36; c++) max = Math.max(max, Math.abs(a[d][k][r][c] - b[d][k][r][c]));
    return max;
  }
  const impactLayers = {
    heatDays: {name: 'Heat days', unit: 'days', colors: ['#242e49', '#986553', '#f3bf75']},
    cumulativeRain: {name: 'Cumulative rain', unit: 'mm', colors: ['#172f43', '#398baf', '#a4dac5']},
    rainDeficit: {name: 'Rain-deficit index', unit: 'index', colors: ['#64b9b1', '#ded8b7', '#df956a']},
    heavyRainDays: {name: 'Heavy-rain days', unit: 'days', colors: ['#263652', '#818bb9', '#d1b9e5']}
  };
  const impactDefaults = {heatC: 40, need: 4, heavyRainMm: 64.5};
  function impactThresholds(options = {}) {
    const thresholds = {...impactDefaults, ...options};
    if (!Number.isFinite(thresholds.heatC) || thresholds.heatC < -100 || thresholds.heatC > 100) throw new Error('Heat threshold must be between −100 and 100 °C.');
    if (!Number.isFinite(thresholds.need) || thresholds.need <= 0) throw new Error('Daily rain need must be greater than zero.');
    if (!Number.isFinite(thresholds.heavyRainMm) || thresholds.heavyRainMm <= 0) throw new Error('Heavy-rain threshold must be greater than zero.');
    return thresholds;
  }
  // Only forecast t2m (K) and tp enter these metrics. The mask selects cells;
  // no initial states, shock magnitudes, crop parameters, or synthetic fields enter.
  function computeLandImpacts(rollout, landMask, options = {}) {
    const thresholds = impactThresholds(options);
    if (!Array.isArray(rollout) || !rollout.length || !M.shape(rollout, [rollout.length, 6, 32, 36])) throw new Error('Impacts require a finite, nonempty rollout shaped [steps,6,32,36].');
    if (!M.shape(landMask, [32, 36])) throw new Error('Land-only impacts require a finite 32×36 land mask.');
    const steps = rollout.length, fields = {}, mean = {};
    for (const key of Object.keys(impactLayers)) fields[key] = M.matrix(() => null);
    let landCells = 0;
    for (let r = 0; r < 32; r++) for (let c = 0; c < 36; c++) {
      if (landMask[r][c] < .5) continue;
      landCells++;
      let heatDays = 0, cumulativeRain = 0, heavyRainDays = 0;
      for (const day of rollout) {
        // Compare in Kelvin to preserve exact equality at a Celsius threshold.
        if (day[0][r][c] > thresholds.heatC + 273.15) heatDays++;
        const rain = day[1][r][c];
        if (rain < 0) throw new Error('Rollout contains negative precipitation.');
        cumulativeRain += rain;
        if (rain >= thresholds.heavyRainMm) heavyRainDays++;
      }
      fields.heatDays[r][c] = heatDays;
      fields.cumulativeRain[r][c] = cumulativeRain;
      // Intentionally unbounded below: a negative index indicates rain surplus.
      fields.rainDeficit[r][c] = 1 - cumulativeRain / (thresholds.need * steps);
      fields.heavyRainDays[r][c] = heavyRainDays;
    }
    for (const key of Object.keys(impactLayers)) {
      mean[key] = landCells ? fields[key].flat().reduce((sum, value) => sum + (value === null ? 0 : value), 0) / landCells : null;
    }
    return {fields, mean, landCells, steps, thresholds};
  }
  function compareLandImpacts(primeRollout, branchRollout, landMask, options = {}) {
    if (primeRollout.length !== branchRollout.length) throw new Error('Prime and Branch must have the same forecast horizon.');
    const prime = computeLandImpacts(primeRollout, landMask, options), branch = computeLandImpacts(branchRollout, landMask, options);
    const delta = {fields: {}, mean: {}};
    for (const key of Object.keys(impactLayers)) {
      delta.fields[key] = M.matrix((r, c) => prime.fields[key][r][c] === null ? null : branch.fields[key][r][c] - prime.fields[key][r][c]);
      delta.mean[key] = prime.landCells ? branch.mean[key] - prime.mean[key] : null;
    }
    return {prime, branch, delta};
  }
  const impactNumber = value => Number(value.toFixed(3)).toLocaleString(undefined, {maximumFractionDigits: 3});
  function impactPolicySentences(comparison) {
    if (!comparison.prime.landCells) return [];
    const {prime, branch, delta} = comparison, t = prime.thresholds;
    const change = (value, unit) => value === 0 ? `no change in mean ${unit}` : `${impactNumber(Math.abs(value))} ${value > 0 ? 'more' : 'fewer'} ${unit} per land cell`;
    const deficit = branch.mean.rainDeficit;
    return [
      `Indicative heat-preparedness cue: over ${prime.steps} forecast days, Branch shows ${change(delta.mean.heatDays, `days above ${t.heatC} °C`)} versus Prime. Use the heat layer to frame a review of exposure preparedness.`,
      `Indicative rainfall-planning cue: mean cumulative rain is ${impactNumber(prime.mean.cumulativeRain)} mm in Prime and ${impactNumber(branch.mean.cumulativeRain)} mm in Branch. At a scenario need of ${t.need} mm/day, Branch's mean deficit index is ${impactNumber(deficit)} (${deficit > 0 ? 'shortfall relative to the chosen need' : deficit < 0 ? 'surplus relative to the chosen need' : 'matches the chosen need'}). Review water-contingency assumptions against the selected need.`,
      `Indicative heavy-rain preparedness cue: Branch shows ${change(delta.mean.heavyRainDays, `days at or above ${t.heavyRainMm} mm`)} versus Prime. Use the heavy-rain layer to frame drainage and flood-readiness discussions.`
    ];
  }
  function createImpactPanel(getContext) {
    const $ = id => document.getElementById(id);
    let cachedResult, cachedMask, cachedSettings, comparison = null, hover = null;
    const signed = value => (value > 0 ? '+' : '') + impactNumber(value);
    function thresholdsFromUI() {
      const ids = ['impact-heat', 'impact-need', 'impact-heavy'];
      if (ids.some(id => !$(id).value.trim())) throw new Error('Enter all three indicative thresholds.');
      return impactThresholds({heatC: Number($('impact-heat').value), need: Number($('impact-need').value),
        heavyRainMm: Number($('impact-heavy').value)});
    }
    function render() {
      const {result, grid, centre, radius} = getContext();
      let message = '', thresholds;
      try {thresholds = thresholdsFromUI();} catch (error) {message = error.message;}
      if (!message && !result) message = 'Run both timelines to calculate indicative impacts from their returned forecasts.';
      if (!message && !grid.mask) message = 'Land-only impacts unavailable: load grid.json with land_sea_mask. Ocean cells are never treated as land.';
      if (!message) {
        const key = JSON.stringify(thresholds);
        try {
          if (result !== cachedResult || grid.mask !== cachedMask || key !== cachedSettings) {
            comparison = compareLandImpacts(result.prime, result.branch, grid.mask, thresholds);
            cachedResult = result; cachedMask = grid.mask; cachedSettings = key;
          }
          if (!comparison.prime.landCells) message = 'The supplied land mask contains no land cells (mask ≥ 0.5).';
        } catch (error) {message = error.message;}
      }
      $('impact-status').textContent = message || `Indicative only · all ${comparison.prime.steps} returned forecast days · ${comparison.prime.landCells} land cells · rain in mm/day.`;
      $('impact-content').hidden = Boolean(message);
      if (message) {comparison = null; cachedResult = null; return;}
      for (const key of Object.keys(impactLayers)) {
        for (const timeline of ['prime', 'branch', 'delta']) {
          $(`impact-${key}-${timeline}`).textContent = (timeline === 'delta' ? signed : impactNumber)(comparison[timeline].mean[key]);
        }
      }
      $('impact-heat-rule').textContent = `t2m > ${thresholds.heatC} °C`;
      $('impact-heavy-rule').textContent = `tp ≥ ${thresholds.heavyRainMm} mm/day`;
      const layer = $('impact-layer').value, definition = impactLayers[layer], timeline = $('impact-timeline').value;
      const values = [...comparison.prime.fields[layer].flat(), ...comparison.branch.fields[layer].flat()].filter(Number.isFinite);
      let lo = Math.min(...values), hi = Math.max(...values);
      if (layer === 'heatDays' || layer === 'heavyRainDays') {lo = 0; hi = comparison.prime.steps;}
      else if (layer === 'cumulativeRain') {lo = 0; hi = Math.max(hi, 1);}
      else {lo = Math.min(lo, 0); hi = Math.max(hi, 1);}
      const deltaValues = comparison.delta.fields[layer].flat().filter(Number.isFinite);
      const limit = Math.max(...deltaValues.map(Math.abs), .01);
      M.drawMap($('impact-map'), grid, comparison[timeline].fields[layer], [lo, hi], definition.colors, centre, radius, hover);
      M.drawMap($('impact-delta-map'), grid, comparison.delta.fields[layer], [-limit, limit], M.config.differenceColors, centre, radius, hover);
      $('impact-map-title').textContent = `Indicative ${timeline === 'prime' ? 'Prime' : 'Branch'} · ${definition.name}`;
      $('impact-delta-title').textContent = `Indicative delta · ${definition.name}`;
      function legend(id, scale, colors) {
        const element = $(id); element.replaceChildren();
        const bar = document.createElement('div'); bar.className = 'legend-bar'; bar.style.background = `linear-gradient(90deg,${colors.join(',')})`;
        const labels = document.createElement('div'); labels.className = 'legend-labels';
        for (const value of [scale[0], (scale[0] + scale[1]) / 2, scale[1]]) {const label = document.createElement('span'); label.textContent = `${impactNumber(value)} ${definition.unit}`; labels.append(label);}
        element.append(bar, labels);
      }
      legend('impact-map-legend', [lo, hi], definition.colors);
      legend('impact-delta-legend', [-limit, limit], M.config.differenceColors);
      $('impact-hover').textContent = !hover ? 'Hover a land cell to inspect Prime, Branch and delta. Dark cells are excluded ocean.'
        : grid.mask[hover.r][hover.c] < .5 ? 'Ocean cell · excluded from indicative impacts.'
        : `Land cell [${hover.r}, ${hover.c}] · Prime ${impactNumber(comparison.prime.fields[layer][hover.r][hover.c])} · Branch ${impactNumber(comparison.branch.fields[layer][hover.r][hover.c])} · Δ ${signed(comparison.delta.fields[layer][hover.r][hover.c])} ${definition.unit}`;
      const policy = $('impact-policy'); policy.replaceChildren();
      for (const sentence of impactPolicySentences(comparison)) {const item = document.createElement('li'); item.textContent = sentence; policy.append(item);}
    }
    for (const id of ['impact-heat', 'impact-need', 'impact-heavy']) $(id).addEventListener('input', render);
    for (const id of ['impact-layer', 'impact-timeline']) $(id).addEventListener('change', render);
    for (const id of ['impact-map', 'impact-delta-map']) {
      const canvas = $(id);
      canvas.addEventListener('mousemove', event => {hover = M.pickCell(canvas, getContext().grid, event); render();});
      canvas.addEventListener('mouseleave', () => {hover = null; render();});
    }
    return {render};
  }
  function drawChart(canvas, values, activeDay, color, unit, band = null) {
    const {ctx, w, h} = M.canvasContext(canvas), pad = {l: 39, r: 12, t: 10, b: 25};
    const plotW = Math.max(1, w - pad.l - pad.r), plotH = h - pad.t - pad.b;
    ctx.clearRect(0, 0, w, h); ctx.font = '10px system-ui';
    if (!values || !values.length) {ctx.fillStyle = '#97a9b7'; ctx.textAlign = 'center'; ctx.fillText('Run both timelines to see the response', w / 2, h / 2); return;}
    const minimum = band ? Math.min(...band.map(day => day.low)) : 0;
    const maximum = band ? Math.max(...band.map(day => day.high)) : Math.max(...values, 0);
    const span = Math.max(maximum - minimum, band ? .05 : 1);
    const x = i => pad.l + (values.length === 1 ? plotW / 2 : i / (values.length - 1) * plotW);
    const y = value => pad.t + plotH - (value - minimum) / span * plotH;
    for (let i = 0; i <= 2; i++) {
      const value = minimum + span * i / 2, yy = y(value);
      ctx.strokeStyle = '#2a3945'; ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
      ctx.fillStyle = '#97a9b7'; ctx.textAlign = 'right'; ctx.fillText(value >= 100 ? Math.round(value).toString() : Number(value.toPrecision(2)).toString(), pad.l - 6, yy + 3);
    }
    ctx.beginPath();
    if (band) {
      if (band.length === 1) {
        const top = y(band[0].high), bottom = y(band[0].low);
        ctx.rect(x(0) - 4, top, 8, Math.max(1, bottom - top));
      } else {
        band.forEach((day, i) => i ? ctx.lineTo(x(i), y(day.high)) : ctx.moveTo(x(i), y(day.high)));
        for (let i = band.length - 1; i >= 0; i--) ctx.lineTo(x(i), y(band[i].low));
        ctx.closePath();
      }
    } else {
      ctx.moveTo(x(0), y(0)); values.forEach((v, i) => ctx.lineTo(x(i), y(v))); ctx.lineTo(x(values.length - 1), y(0)); ctx.closePath();
    }
    ctx.fillStyle = color + (band ? '44' : '14'); ctx.fill();
    ctx.beginPath(); values.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    ctx.strokeStyle = '#647e8a'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x(activeDay), pad.t); ctx.lineTo(x(activeDay), y(minimum)); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x(activeDay), y(values[activeDay]), 4, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    ctx.fillStyle = '#97a9b7'; ctx.textAlign = 'left'; ctx.fillText('Day 1', pad.l, h - 5); ctx.textAlign = 'right';
    if (values.length > 1) ctx.fillText(`Day ${values.length}`, w - pad.r, h - 5);
    canvas.setAttribute('aria-label', `${unit} by day: ${values.map((v, i) => {
      const range = band ? `, range ${band[i].low.toPrecision(4)} to ${band[i].high.toPrecision(4)}` : '';
      return `${i + 1}: ${v.toPrecision(4)}${range}`;
    }).join(', ')}`);
  }
  Object.assign(M, {responseMetrics, maxAbsDiff, drawChart, impactLayers, impactDefaults, impactThresholds,
    computeLandImpacts, compareLandImpacts, impactPolicySentences, createImpactPanel});
})(window.Monsoon);
