/**
 * Signifier Panel Module
 *
 * Displays the 30 expression signifier scores (0-100) in four groups beneath
 * the avatar chat. Each metric shows a label, a mini bar, and the value.
 */

var SIGNIFIER_LABELS = {
    g1_duchenne: 'Duchenne Marker',
    g1_pupil_dilation: 'Pupil Dilation',
    g1_eyebrow_flash: 'Eyebrow Flash',
    g1_eye_contact: 'Sustained Eye Contact',
    g1_head_tilt: 'Head Tilt (Lateral)',
    g1_forward_lean: 'Forward Lean',
    g1_mirroring: 'Mirroring',
    g1_rhythmic_nodding: 'Rhythmic Nodding',
    g1_parted_lips: 'Parted Lips',
    g1_softened_forehead: 'Softened Forehead',
    g2_look_up_lr: 'Look Up Left/Right',
    g2_lip_pucker: 'Lip Pucker',
    g2_eye_squint: 'Eye Squinting',
    g2_thinking_brow: 'Thinking Brow',
    g2_chin_stroke: 'Chin Stroke',
    g2_stillness: 'Stillness',
    g2_lowered_brow: 'Lowered Brow',
    g3_contempt: 'Contempt',
    g3_nose_crinkle: 'Nose Crinkle',
    g3_lip_compression: 'Lip Compression',
    g3_eye_block: 'Eye Block',
    g3_jaw_clench: 'Jaw Clenching',
    g3_rapid_blink: 'Rapid Blinking',
    g3_gaze_aversion: 'Gaze Aversion',
    g3_no_nod: 'No-Nod',
    g3_narrowed_pupils: 'Narrowed Pupils',
    g3_mouth_cover: 'Mouth Cover',
    g4_relaxed_exhale: 'Relaxed Exhale',
    g4_fixed_gaze: 'Fixed Gaze',
    g4_smile_transition: 'Smile to Genuine'
};

var SIGNIFIER_GROUPS = [
    { id: 'g1', title: 'Interest & Engagement', keys: ['g1_duchenne', 'g1_pupil_dilation', 'g1_eyebrow_flash', 'g1_eye_contact', 'g1_head_tilt', 'g1_forward_lean', 'g1_mirroring', 'g1_rhythmic_nodding', 'g1_parted_lips', 'g1_softened_forehead'] },
    { id: 'g2', title: 'Cognitive Load', keys: ['g2_look_up_lr', 'g2_lip_pucker', 'g2_eye_squint', 'g2_thinking_brow', 'g2_chin_stroke', 'g2_stillness', 'g2_lowered_brow'] },
    { id: 'g3', title: 'Resistance & Objections', keys: ['g3_contempt', 'g3_nose_crinkle', 'g3_lip_compression', 'g3_eye_block', 'g3_jaw_clench', 'g3_rapid_blink', 'g3_gaze_aversion', 'g3_no_nod', 'g3_narrowed_pupils', 'g3_mouth_cover'] },
    { id: 'g4', title: 'Decision-Ready', keys: ['g4_relaxed_exhale', 'g4_fixed_gaze', 'g4_smile_transition'] }
];

function getColorForScore(score) {
    if (score <= 50) {
        var r = 255, g = Math.round(255 * (score / 50)), b = 0;
        return 'rgb(' + r + ',' + g + ',' + b + ')';
    }
    var r = Math.round(255 * (1 - (score - 50) / 50)), g = 255, b = 0;
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

class SignifierPanel {
    constructor(options) {
        this.containerId = options.containerId || 'signifierPanelContainer';
        this.container = null;
        this.rows = {};
    }

    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) return;

        var inner = document.createElement('div');
        inner.className = 'signifier-panel-inner';

        var header = document.createElement('div');
        header.className = 'signifier-panel-header';
        header.textContent = 'Engagement Metrics (30 signifiers)';
        inner.appendChild(header);

        var groupsEl = document.createElement('div');
        groupsEl.className = 'signifier-panel-groups';

        for (var g = 0; g < SIGNIFIER_GROUPS.length; g++) {
            var gr = SIGNIFIER_GROUPS[g];
            var groupDiv = document.createElement('div');
            groupDiv.className = 'signifier-group';
            groupDiv.setAttribute('data-group', gr.id);

            var title = document.createElement('div');
            title.className = 'signifier-group-title';
            title.textContent = gr.title;
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

    update(scores) {
        if (!scores || typeof scores !== 'object') {
            this.reset();
            return;
        }
        for (var key in this.rows) {
            var r = this.rows[key];
            var v = scores[key];
            var num = (typeof v === 'number' && isFinite(v)) ? Math.max(0, Math.min(100, v)) : null;
            if (num !== null) {
                r.fill.style.width = num + '%';
                r.fill.style.background = getColorForScore(num);
                r.value.textContent = Math.round(num);
            } else {
                r.fill.style.width = '0%';
                r.value.textContent = '--';
            }
        }
    }

    reset() {
        for (var key in this.rows) {
            var r = this.rows[key];
            r.fill.style.width = '0%';
            r.value.textContent = '--';
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SignifierPanel;
}
