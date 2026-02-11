/**
 * Azure Metrics Panel Module
 *
 * Displays Azure Face API engagement metrics on a continuous 0-100 scale when detection
 * method is Azure Face API: base emotions and B2B composite features. Each metric shows
 * a label, a bar (width = score%), and the numeric score. Color bands: 0-39 muted,
 * 40-69 medium, 70-100 high. No "detected" / "not detected"; all values are continuous.
 */

var AZURE_BASE_LABELS = {
    anger: 'Anger',
    contempt: 'Contempt',
    disgust: 'Disgust',
    fear: 'Fear',
    happiness: 'Happiness',
    neutral: 'Neutral',
    sadness: 'Sadness',
    surprise: 'Surprise'
};

var AZURE_COMPOSITE_LABELS = {
    receptive: 'Receptive',
    focused: 'Focused',
    interested: 'Interested',
    agreeable: 'Agreeable',
    open: 'Open',
    skeptical: 'Skeptical',
    concerned: 'Concerned',
    disagreeing: 'Disagreeing',
    stressed: 'Stressed',
    disengaged: 'Disengaged'
};

var AZURE_GROUPS = [
    { id: 'base', title: 'Emotions (Azure)', keys: ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'] },
    { id: 'composite', title: 'B2B Composites', keys: ['receptive', 'focused', 'interested', 'agreeable', 'open', 'skeptical', 'concerned', 'disagreeing', 'stressed', 'disengaged'] }
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

/** Real-time metric display; metrics are continuous 0-100, bar width = score%. */

class AzureMetricsPanel {
    constructor(options) {
        this.containerId = options.containerId || 'azureMetricsPanelContainer';
        this.container = null;
        this.rows = {};
        this.rawValues = {};
    }

    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) return;

        var inner = document.createElement('div');
        inner.className = 'signifier-panel-inner azure-metrics-panel-inner';

        var header = document.createElement('div');
        header.className = 'signifier-panel-header';
        header.textContent = 'Azure metrics';
        inner.appendChild(header);

        var groupsEl = document.createElement('div');
        groupsEl.className = 'signifier-panel-groups';

        for (var g = 0; g < AZURE_GROUPS.length; g++) {
            var gr = AZURE_GROUPS[g];
            var groupDiv = document.createElement('div');
            groupDiv.className = 'signifier-group';
            groupDiv.setAttribute('data-group', gr.id);

            var title = document.createElement('div');
            title.className = 'signifier-group-title';
            title.textContent = gr.title;
            groupDiv.appendChild(title);

            var list = document.createElement('div');
            list.className = 'signifier-list';

            var labels = gr.id === 'base' ? AZURE_BASE_LABELS : AZURE_COMPOSITE_LABELS;
            for (var k = 0; k < gr.keys.length; k++) {
                var key = gr.keys[k];
                var row = document.createElement('div');
                row.className = 'signifier-row';
                row.setAttribute('data-key', key);

                var label = document.createElement('span');
                label.className = 'signifier-label';
                label.textContent = labels[key] || key;
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
            groupDiv.appendChild(list);
            groupsEl.appendChild(groupDiv);
        }
        inner.appendChild(groupsEl);
        this.container.appendChild(inner);
    }

    /**
     * Update display from continuous 0-100 Azure metrics (base + composite).
     * Bar width = score%; value = numeric score (rounded); color by bands (low/medium/high).
     * @param {Object} azureMetrics - { base: {}, composite: {} }; values 0-100 (continuous).
     */
    update(azureMetrics) {
        if (!azureMetrics || typeof azureMetrics !== 'object') {
            this.reset();
            return;
        }
        var base = azureMetrics.base && typeof azureMetrics.base === 'object' ? azureMetrics.base : {};
        var composite = azureMetrics.composite && typeof azureMetrics.composite === 'object' ? azureMetrics.composite : {};
        var allScores = Object.assign({}, base, composite);

        for (var key in this.rows) {
            var r = this.rows[key];
            var v = allScores[key];
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
    module.exports = AzureMetricsPanel;
}
