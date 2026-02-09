/**
 * Composite Metrics Panel Module
 *
 * Displays facial+speech composite metrics (0-100 continuous) when detection
 * is MediaPipe or unified. Composites combine group means (G1-G4) with recent speech
 * tags (commitment, interest, confusion, objection, etc.). Each metric shows a label,
 * a bar (width = score%), and the numeric score. Color bands: 0-39 muted, 40-69 medium,
 * 70-100 high. For disengagement_risk_multimodal, high score = high risk (re-engage).
 */

var COMPOSITE_LABELS = {
    verbal_nonverbal_alignment: 'Verbal–nonverbal alignment',
    cognitive_load_multimodal: 'Cognitive load (multimodal)',
    rapport_engagement: 'Rapport / engagement',
    skepticism_objection_strength: 'Skepticism / objection strength',
    decision_readiness_multimodal: 'Decision readiness (multimodal)',
    opportunity_strength: 'Opportunity strength (closing moment)',
    trust_rapport: 'Trust / rapport',
    disengagement_risk_multimodal: 'Disengagement risk (multimodal)',
    confusion_multimodal: 'Confusion (multimodal)',
    tension_objection_multimodal: 'Tension / objection (multimodal)',
    loss_of_interest_multimodal: 'Loss of interest (multimodal)',
    decision_plus_voice: 'Decision readiness + voice',
    psychological_safety_proxy: 'Psychological safety proxy',
    urgency_sensitivity: 'Urgency sensitivity',
    skepticism_strength: 'Skepticism strength',
    enthusiasm_multimodal: 'Enthusiasm (multimodal)',
    hesitation_multimodal: 'Hesitation (multimodal)',
    authority_deferral: 'Authority deferral',
    rapport_depth: 'Rapport depth',
    cognitive_overload_proxy: 'Cognitive overload proxy'
};

var COMPOSITE_KEYS = [
    'verbal_nonverbal_alignment',
    'cognitive_load_multimodal',
    'rapport_engagement',
    'skepticism_objection_strength',
    'decision_readiness_multimodal',
    'opportunity_strength',
    'trust_rapport',
    'disengagement_risk_multimodal',
    'confusion_multimodal',
    'tension_objection_multimodal',
    'loss_of_interest_multimodal',
    'decision_plus_voice',
    'psychological_safety_proxy',
    'urgency_sensitivity',
    'skepticism_strength',
    'enthusiasm_multimodal',
    'hesitation_multimodal',
    'authority_deferral',
    'rapport_depth',
    'cognitive_overload_proxy'
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
     * Bar width = score%; value = numeric score (rounded); color by bands (low/medium/high).
     * @param {Object} compositeMetrics - Keyed by composite id; values 0-100 (continuous).
     */
    update(compositeMetrics) {
        if (!compositeMetrics || typeof compositeMetrics !== 'object') {
            this.reset();
            return;
        }
        for (var i = 0; i < COMPOSITE_KEYS.length; i++) {
            var key = COMPOSITE_KEYS[i];
            var r = this.rows[key];
            var v = compositeMetrics[key];
            var num = (typeof v === 'number' && isFinite(v)) ? v : (typeof v === 'string' ? parseFloat(v) : NaN);
            var normalized = (!isNaN(num)) ? Math.max(0, Math.min(100, num)) : NaN;
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
