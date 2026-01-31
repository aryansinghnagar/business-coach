/**
 * Azure Metrics Panel Module
 *
 * Displays Azure Face API engagement metrics when detection method is Azure Face API:
 * - Base emotions (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise)
 * - B2B composite features (receptive, focused, interested, agreeable, open, skeptical,
 *   concerned, disagreeing, stressed, disengaged).
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

function getColorForScore(score) {
    if (score <= 30) {
        var t = score / 30;
        var r = Math.round(239 * (1 - t) + 167 * t);
        var g = Math.round(68 * (1 - t) + 139 * t);
        var b = Math.round(68 * (1 - t) + 250 * t);
        return 'rgb(' + r + ',' + g + ',' + b + ')';
    }
    if (score <= 70) {
        var t = (score - 30) / 40;
        var r = Math.round(167 + (34 - 167) * t);
        var g = Math.round(139 + (197 - 139) * t);
        var b = Math.round(250 + (94 - 250) * t);
        return 'rgb(' + r + ',' + g + ',' + b + ')';
    }
    var t = (score - 70) / 30;
    var r = Math.round(34 + (16 - 34) * t);
    var g = Math.round(197 + (185 - 197) * t);
    var b = Math.round(94 + (129 - 94) * t);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

/** Real-time metric display (no smoothing). */

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
        header.textContent = 'Azure Face API – Emotions & B2B Composites';
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
            var raw = (!isNaN(num)) ? Math.max(0, Math.min(100, num)) : null;
            if (raw !== null) {
                var display = Math.max(0, Math.min(100, raw));
                r.fill.style.width = display + '%';
                r.fill.style.background = getColorForScore(display);
                r.value.textContent = Math.round(display);
            } else {
                r.fill.style.width = '0%';
                r.value.textContent = '--';
            }
        }
    }

    reset() {
        this.rawValues = {};
        for (var key in this.rows) {
            var r = this.rows[key];
            r.fill.style.width = '0%';
            r.value.textContent = '--';
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AzureMetricsPanel;
}
