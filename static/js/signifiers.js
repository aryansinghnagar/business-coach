"use strict";
// ========================================================================
// SIGNIFIER PANEL (signifier-panel.js)
// ========================================================================
//
// WHAT THIS MODULE DOES (in plain language):
// ------------------------------------------
// Shows the expression signifier scores (0–100) in four groups under the chat.
// Each row has a label, a bar whose width is the score percent, and the number.
// Colors: 0–39 muted, 40–69 medium, 70–100 high. All values are continuous
// (no "detected" / "not detected").
//
// WHO USES THIS: app-main.js creates the panel; engagement-detector.js (or the
// main app) pushes state updates so this panel can refresh the bars in real time.
//

var SIGNIFIER_LABELS = {
    g1_duchenne: 'Duchenne Marker',
    g1_pupil_dilation: 'Pupil Dilation',
    g1_eyebrow_flash: 'Eyebrow Flash',
    g1_eye_contact: 'Sustained Eye Contact',
    g1_head_tilt: 'Head Tilt (Lateral)',
    g1_forward_lean: 'Forward Lean',
    g1_facial_symmetry: 'Facial Symmetry',
    g1_rhythmic_nodding: 'Rhythmic Nodding',
    g1_parted_lips: 'Parted Lips',
    g1_softened_forehead: 'Softened Forehead',
    g1_micro_smile: 'Micro Smile',
    g1_brow_raise_sustained: 'Brow Raise Sustained',
    g1_mouth_open_receptive: 'Mouth Open Receptive',
    g1_eye_widening: 'Eye Widening',
    g1_nod_intensity: 'Nod Intensity',
    g2_look_up_lr: 'Look Up Left/Right',
    g2_lip_pucker: 'Lip Pucker',
    g2_eye_squint: 'Eye Squinting',
    g2_thinking_brow: 'Thinking Brow',
    g2_stillness: 'Stillness',
    g2_gaze_shift_frequency: 'Gaze Shift Frequency',
    g2_mouth_tight_eval: 'Mouth Tight (Evaluation)',
    g3_contempt: 'Contempt',
    g3_nose_crinkle: 'Nose Crinkle',
    g3_lip_compression: 'Lip Compression',
    g3_eye_block: 'Eye Block',
    g3_jaw_clench: 'Jaw Clenching',
    g3_rapid_blink: 'Rapid Blinking',
    g3_gaze_aversion: 'Gaze Aversion',
    g3_no_nod: 'No-Nod',
    g3_narrowed_pupils: 'Narrowed Pupils',
    g3_lip_corner_dip: 'Lip Corner Dip',
    g3_brow_lower_sustained: 'Brow Lower Sustained',
    g3_eye_squeeze: 'Eye Squeeze',
    g3_head_shake: 'Head Shake',
    g4_relaxed_exhale: 'Relaxed Exhale',
    g4_fixed_gaze: 'Fixed Gaze',
    g4_smile_transition: 'Smile to Genuine',
    g4_mouth_relax: 'Mouth Relax',
    g4_smile_sustain: 'Smile Sustain'
};

var SIGNIFIER_GROUPS = [
    { id: 'g1', title: 'Interest & Engagement', keys: ['g1_duchenne', 'g1_pupil_dilation', 'g1_eyebrow_flash', 'g1_eye_contact', 'g1_head_tilt', 'g1_forward_lean', 'g1_facial_symmetry', 'g1_rhythmic_nodding', 'g1_parted_lips', 'g1_softened_forehead', 'g1_micro_smile', 'g1_brow_raise_sustained', 'g1_mouth_open_receptive', 'g1_eye_widening', 'g1_nod_intensity'] },
    { id: 'g2', title: 'Cognitive Load', keys: ['g2_look_up_lr', 'g2_lip_pucker', 'g2_eye_squint', 'g2_thinking_brow', 'g2_stillness', 'g2_gaze_shift_frequency', 'g2_mouth_tight_eval'] },
    { id: 'g3', title: 'Resistance & Objections', keys: ['g3_contempt', 'g3_nose_crinkle', 'g3_lip_compression', 'g3_eye_block', 'g3_jaw_clench', 'g3_rapid_blink', 'g3_gaze_aversion', 'g3_no_nod', 'g3_narrowed_pupils', 'g3_lip_corner_dip', 'g3_brow_lower_sustained', 'g3_eye_squeeze', 'g3_head_shake'] },
    { id: 'g4', title: 'Decision-Ready', keys: ['g4_relaxed_exhale', 'g4_fixed_gaze', 'g4_smile_transition', 'g4_mouth_relax', 'g4_smile_sustain'] }
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

/** Real-time metric display; scores are continuous 0-100, bar width = score%. */

class SignifierPanel {
    constructor(options) {
        this.containerId = options.containerId || 'signifierPanelContainer';
        this.container = null;
        this.rows = {};
        /** Per-signifier raw values (0–100) for real-time display. */
        this.rawValues = {};
        /** Throttle: max DOM updates per second (performance). */
        this._throttleMs = 250;
        this._lastUpdateTime = 0;
    }

    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) return;

        var inner = document.createElement('div');
        inner.className = 'signifier-panel-inner';

        var header = document.createElement('div');
        header.className = 'signifier-panel-header';
        header.textContent = '';
        inner.appendChild(header);

        var groupsEl = document.createElement('div');
        groupsEl.className = 'signifier-panel-groups';

        for (var g = 0; g < SIGNIFIER_GROUPS.length; g++) {
            var gr = SIGNIFIER_GROUPS[g];
            var groupDiv = document.createElement('div');
            groupDiv.className = 'signifier-group';
            groupDiv.setAttribute('data-group', gr.id);

            var title = document.createElement('div');
            title.className = 'signifier-group-title collapsible';
            title.setAttribute('role', 'button');
            title.setAttribute('aria-expanded', 'true');
            title.setAttribute('tabindex', '0');
            title.appendChild(document.createTextNode(gr.title + ' '));
            var chevron = document.createElement('span');
            chevron.className = 'chevron';
            chevron.setAttribute('aria-hidden', 'true');
            chevron.textContent = '\u25BC';
            title.appendChild(chevron);
            (function (groupEl, titleEl) {
                title.addEventListener('click', function () {
                    var collapsed = groupEl.classList.toggle('collapsed');
                    titleEl.classList.toggle('collapsed', collapsed);
                    titleEl.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
                });
                title.addEventListener('keydown', function (ev) {
                    if (ev.key === 'Enter' || ev.key === ' ') {
                        ev.preventDefault();
                        title.click();
                    }
                });
            }(groupDiv, title));
            groupDiv.appendChild(title);

            var list = document.createElement('div');
            list.className = 'signifier-list';

            for (var k = 0; k < gr.keys.length; k++) {
                var key = gr.keys[k];
                var row = document.createElement('div');
                row.className = 'signifier-row';
                row.setAttribute('data-key', key);

                var label = document.createElement('span');
                label.className = 'signifier-label';
                label.textContent = SIGNIFIER_LABELS[key] || key;
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
     * Update display from continuous 0-100 signifier scores.
     * Throttled and only updates DOM when values changed (reduces reflows).
     * @param {Object} scores - Keyed by signifier id; values 0-100 (continuous).
     */
    update(scores) {
        if (!scores || typeof scores !== 'object') {
            this.reset();
            return;
        }
        var now = Date.now();
        if (now - this._lastUpdateTime < this._throttleMs) {
            return;
        }
        var changed = false;
        for (var key in this.rows) {
            var r = this.rows[key];
            var v = scores[key];
            var num = (typeof v === 'number' && isFinite(v)) ? v : (typeof v === 'string' ? parseFloat(v) : NaN);
            var normalized = (!isNaN(num)) ? Math.max(0, Math.min(100, num)) : NaN;
            var prev = this.rawValues[key];
            if (prev === normalized) continue;
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
    module.exports = SignifierPanel;
}
