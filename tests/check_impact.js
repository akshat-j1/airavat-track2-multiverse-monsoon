#!/usr/bin/env node
/* Numerical checks for rollout-only, land-only indicative impact metrics.
 * Run: node tests/check_impact.js (no npm dependencies or server required).
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = {window: {Monsoon: {config: {height: 32, width: 36}}}};
for (const file of ['core.js', 'impact.js', 'ensemble.js']) {
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '..', 'dashboard', file), 'utf8'), context, {filename: file});
}
const M = context.window.Monsoon;
let failures = 0;
function check(name, fn) {
  try {fn(); console.log('PASS ' + name);} catch (error) {failures++; console.error('FAIL ' + name + ': ' + error.message);}
}
const mask = M.matrix(() => 0);
mask[0][0] = 1; mask[0][1] = .5;
const rollout = (temperatures, rains) => temperatures.map((temperature, d) =>
  Array.from({length: 6}, (_, k) => M.matrix(() => k === 0 ? temperature : k === 1 ? rains[d] : 0)));
const prime = rollout([313.15, 314.15, 312.15], [0, 64.5, 3.5]);
// Second land cell stays cool and dry; all sea cells are extremely hot and wet.
for (const day of prime) {
  day[0][0][1] = 273.15; day[1][0][1] = 0;
  day[0][1][1] = 1000; day[1][1][1] = 100000;
}
const result = M.computeLandImpacts(prime, mask);
check('strict heat threshold, inclusive heavy-rain threshold, cumulative rain', () => {
  assert.equal(result.fields.heatDays[0][0], 1);
  assert.equal(result.fields.heavyRainDays[0][0], 1);
  assert.equal(result.fields.cumulativeRain[0][0], 68);
});
check('exact deficit formula uses actual rollout length and preserves surplus', () => {
  assert.equal(result.steps, 3);
  assert.equal(result.fields.rainDeficit[0][0], 1 - 68 / (4 * 3));
  assert(result.fields.rainDeficit[0][0] < 0);
  assert.equal(result.fields.rainDeficit[0][1], 1);
});
check('mask threshold, ocean exclusion and unweighted per-land-cell means', () => {
  assert.equal(result.landCells, 2);
  assert.equal(result.fields.heatDays[1][1], null);
  assert.equal(result.fields.cumulativeRain[1][1], null);
  assert.equal(result.mean.heatDays, .5);
  assert.equal(result.mean.heavyRainDays, .5);
  assert.equal(result.mean.cumulativeRain, 34);
});
check('editable thresholds recompute without changing the rollout', () => {
  const revised = M.computeLandImpacts(prime, mask, {heatC: 39, need: 8, heavyRainMm: 65});
  assert.equal(revised.fields.heatDays[0][0], 2);
  assert.equal(revised.fields.heavyRainDays[0][0], 0);
  assert.equal(revised.fields.rainDeficit[0][0], 1 - 68 / 24);
  assert.equal(prime[0][0][0][0], 313.15);
});
check('metres-to-mm conversion happens only when importing a window', () => {
  const meters = rollout([313.15, 314.15, 313.15], [.0645, .001, 0]);
  const imported = M.parseWindow(meters, 'm');
  const converted = M.computeLandImpacts(imported.init, mask);
  assert.equal(converted.fields.cumulativeRain[0][0], 65.5);
  assert.equal(converted.fields.heavyRainDays[0][0], 1);
  assert.equal(meters[0][1][0][0], .0645);
  assert.equal(M.computeLandImpacts(meters, mask, {rainUnit: 'm'}).fields.heavyRainDays[0][0], 0);
});
const branch = JSON.parse(JSON.stringify(prime));
branch[0][0][0][0] += 2; branch[2][1][0][0] += 1;
const compared = M.compareLandImpacts(prime, branch, mask);
check('delta is Branch minus Prime after computing each nonlinear metric', () => {
  assert.equal(compared.delta.fields.heatDays[0][0], 1);
  assert.equal(compared.delta.mean.heatDays, .5);
  assert.equal(compared.delta.fields.cumulativeRain[0][0], 1);
  assert.equal(compared.delta.fields.heatDays[1][1], null);
  assert.equal(compared.delta.mean.rainDeficit, compared.branch.mean.rainDeficit - compared.prime.mean.rainDeficit);
});
check('zero shock produces exact zero deltas', () => {
  const same = M.compareLandImpacts(prime, prime, mask);
  for (const key of Object.keys(M.impactLayers)) {
    assert.equal(same.delta.mean[key], 0);
    assert.equal(same.delta.fields[key][0][0], 0);
  }
});
check('missing land mask and invalid inputs are rejected', () => {
  assert.throws(() => M.computeLandImpacts(prime, null), /land mask/);
  assert.throws(() => M.computeLandImpacts([], mask), /rollout/);
  assert.throws(() => M.computeLandImpacts(prime, mask, {need: 0}), /greater than zero/);
  assert.throws(() => M.computeLandImpacts(prime, mask, {heatC: NaN}), /Heat threshold/);
  assert.throws(() => M.computeLandImpacts(prime, mask, {heavyRainMm: -1}), /greater than zero/);
  assert.throws(() => M.compareLandImpacts(prime, branch.slice(1), mask), /same forecast horizon/);
});
check('all-ocean mask returns unavailable means, not fabricated zero exposures', () => {
  const ocean = M.computeLandImpacts(prime, M.matrix(() => 0));
  assert.equal(ocean.landCells, 0);
  assert.equal(ocean.mean.heatDays, null);
  assert.equal(ocean.mean.cumulativeRain, null);
});
check('policy templates use computed comparisons and carry indicative labels', () => {
  const sentences = M.impactPolicySentences(compared);
  assert.equal(sentences.length, 3);
  assert(sentences.every(sentence => sentence.startsWith('Indicative')));
  assert(sentences[0].includes('0.5 more days above 40 °C'));
  assert(sentences[1].includes('34.5 mm in Branch'));
});
check('ensemble bands and exceedance probabilities use land cells and real members', () => {
  const first = rollout([313.15], [64.5]);
  const second = rollout([315.15], [63]);
  second[0][0][0][1] = 273.15;
  second[0][1][0][1] = 0;
  const [summary] = M.summarizeEnsemble([first, second], mask, M.impactDefaults);
  assert.equal(summary.heatProbability, .25);
  assert.equal(summary.heavyProbability, .5);
  assert.equal(summary.rain.low, 31.5);
  assert.equal(summary.rain.high, 64.5);
  assert.equal(summary.rain.median, 48);
});
check('ensemble perturbations are localized, small, and respect input bounds', () => {
  const last = rollout([350], [0])[0];
  const draws = [.1, .5, .5, .1, .9, .5];
  const delta = M.sensitivityPerturbation(last, () => draws.shift());
  assert.equal(delta[0][16][18], 0);
  last[0][16][18] = 300;
  const reroll = [.1, .5, .5, .1, .9, .5];
  const shifted = M.sensitivityPerturbation(last, () => reroll.shift());
  assert(shifted[0][16][18] > 0 && shifted[0][16][18] < .8);
  assert.equal(shifted[0][0][0], 0);
  assert.equal(shifted[2][16][18], 0);
});
process.exitCode = failures ? 1 : 0;
