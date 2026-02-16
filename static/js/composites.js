"use strict";
// ========================================================================
// COMPOSITE METRICS PANEL (composite-metrics-panel.js)
// ========================================================================
//
// WHAT THIS MODULE DOES (in plain language):
// ------------------------------------------
// Shows composite metrics (0–100) when we use MediaPipe or unified detection.
// Composites mix face groups (G1–G4) with recent speech tags (e.g. commitment,
// confusion, objection). Each row: label, bar (width = score%), number. Colors:
// 0–39 muted, 40–69 medium, 70–100 high. For disengagement_risk, high = high risk.
//
// WHO USES THIS: app-main.js creates the panel; engagement state pushes updates
// so this panel can refresh the composite bars in real time.
//

var COMPOSITE_LABELS = {
    verbal_nonverbal_alignment: 'Words match demeanor',
    cognitive_load_multimodal: 'Information overload',
    rapport_engagement: 'Rapport with speaker',
    skepticism_objection_strength: 'Objection or pushback',
    decision_readiness_multimodal: 'Ready to decide',
    opportunity_strength: 'Closing opportunity',
    trust_rapport: 'Trust and rapport',
    disengagement_risk_multimodal: 'Attention slipping',
    confusion_multimodal: 'Confusion or uncertainty',
    tension_objection_multimodal: 'Tension or objection',
    loss_of_interest_multimodal: 'Losing interest',
    decision_plus_voice: 'Ready to close (with voice)',
    psychological_safety_proxy: 'Psychological safety',
    urgency_sensitivity: 'Urgency or timeline focus',
    skepticism_strength: 'Skepticism strength',
    enthusiasm_multimodal: 'Enthusiasm',
    hesitation_multimodal: 'Hesitation or hedging',
    authority_deferral: 'Deferring to authority',
    rapport_depth: 'Rapport depth',
    cognitive_overload_proxy: 'Cognitive overload'
};

// Order: meeting-critical first (decision, rapport, objection, confusion, attention), then alignment and dynamics
var COMPOSITE_KEYS = [
    'decision_readiness_multimodal',
    'rapport_engagement',
    'skepticism_objection_strength',
    'confusion_multimodal',
    'disengagement_risk_multimodal',
    'verbal_nonverbal_alignment',
    'opportunity_strength',
    'loss_of_interest_multimodal',
    'tension_objection_multimodal',
    'decision_plus_voice',
    'trust_rapport',
    'cognitive_load_multimodal',
    'psychological_safety_proxy',
    'enthusiasm_multimodal',
    'hesitation_multimodal',
    'urgency_sensitivity',
    'skepticism_strength',
    'rapport_depth',
    'cognitive_overload_proxy',
    'authority_deferral'
];

/**
 * Continuous color by score: 0-39 muted, 40-69 medium, 70-100 high.
 * @param {number} score - 0-100
 * @returns {string} CSS color
 */
function getColorForScore(score) {
    if (score >= 70) return '#22c55e';
    if (score >= 40) return '#eab308';
    return '#64748b';
}

/** Real-time composite metric display; scores are continuous 0-100, bar width = score%. */

class CompositeMetricsPanel {
    constructor(options) {
        this.containerId = options.containerId || 'compositeMetricsPanelContainer';
        this.container = null;
        this.rows = {};
        this.rawValues = {};
        this._throttleMs = 250;
        this._lastUpdateTime = 0;
    }

    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) return;

        var inner = document.createElement('div');
        inner.className = 'signifier-panel-inner composite-metrics-panel-inner';

        var header = document.createElement('div');
        header.className = 'signifier-panel-header';
        header.textContent = 'Composite metrics (facial + speech)';
        inner.appendChild(header);

        var list = document.createElement('div');
        list.className = 'signifier-list';

        for (var i = 0; i < COMPOSITE_KEYS.length; i++) {
            var key = COMPOSITE_KEYS[i];
            var row = document.createElement('div');
            row.className = 'signifier-row';
            row.setAttribute('data-key', key);

            var label = document.createElement('span');
            label.className = 'signifier-label';
            label.textContent = COMPOSITE_LABELS[key] || key;
            row.appendChild(label);

            var barWrap = document.createElement('div');
            barWrap.className = 'signifier-bar';
            var barFill = document.createElement('div');
            barFill.className = 'signifier-bar-fill';
            barWrap.appendChild(barFill);
            row.appendChild(barWrap);

            var val = document.createElement('span');
            val.className = 'signifier-value';
            val.textContent = '--';
            row.appendChild(val);

            list.appendChild(row);
            this.rows[key] = { fill: barFill, value: val };
        }
        inner.appendChild(list);
        this.container.appendChild(inner);
    }

    /**
     * Update display from continuous 0-100 composite metrics.
     * Throttled and only updates DOM when values changed.
     * @param {Object} compositeMetrics - Keyed by composite id; values 0-100 (continuous).
     */
    update(compositeMetrics) {
        if (!compositeMetrics || typeof compositeMetrics !== 'object') {
            this.reset();
            return;
        }
        var now = Date.now();
        if (now - this._lastUpdateTime < this._throttleMs) return;
        var changed = false;
        for (var i = 0; i < COMPOSITE_KEYS.length; i++) {
            var key = COMPOSITE_KEYS[i];
            var r = this.rows[key];
            var v = compositeMetrics[key];
            var num = (typeof v === 'number' && isFinite(v)) ? v : (typeof v === 'string' ? parseFloat(v) : NaN);
            var normalized = (!isNaN(num)) ? Math.max(0, Math.min(100, num)) : NaN;
            if (this.rawValues[key] === normalized) continue;
            changed = true;
            this.rawValues[key] = normalized;
            if (isNaN(normalized)) {
                r.fill.style.width = '0%';
                r.fill.style.background = '#64748b';
                r.value.textContent = '\u2014';
            } else {
                r.fill.style.width = normalized + '%';
                r.fill.style.background = getColorForScore(normalized);
                r.value.textContent = Math.round(normalized).toString();
            }
        }
        if (changed) this._lastUpdateTime = now;
    }

    reset() {
        this.rawValues = {};
        for (var key in this.rows) {
            var r = this.rows[key];
            r.fill.style.width = '0%';
            r.fill.style.background = '#64748b';
            r.value.textContent = '\u2014';
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CompositeMetricsPanel;
}
