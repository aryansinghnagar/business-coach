/**
 * Inline engagement dashboard: drives the left-panel metrics in index.html
 * (replaces the separate dashboard.html page). Only runs when the inline
 * dashboard container is present (main window, not panel-only).
 */
(function () {
    'use strict';

    function getApiBase() {
        if (typeof window !== 'undefined' && window.location && window.location.origin) {
            var o = window.location.origin;
            if (o && (o.indexOf('http://') === 0 || o.indexOf('https://') === 0)) return o;
        }
        return 'http://localhost:5000';
    }

    function barColor(score) {
        if (score >= 70) return '#22c55e';
        if (score >= 40) return '#eab308';
        return '#64748b';
    }

    var SIGNIFIER_LABELS = {
        g1_duchenne: 'Duchenne Marker', g1_pupil_dilation: 'Pupil Dilation', g1_eyebrow_flash: 'Eyebrow Flash',
        g1_eye_contact: 'Sustained Eye Contact', g1_head_tilt: 'Head Tilt (Lateral)', g1_forward_lean: 'Forward Lean',
        g1_facial_symmetry: 'Facial Symmetry', g1_rhythmic_nodding: 'Rhythmic Nodding', g1_parted_lips: 'Parted Lips', g1_softened_forehead: 'Softened Forehead',
        g1_micro_smile: 'Micro Smile', g1_brow_raise_sustained: 'Brow Raise Sustained', g1_mouth_open_receptive: 'Mouth Open Receptive', g1_eye_widening: 'Eye Widening', g1_nod_intensity: 'Nod Intensity',
        g2_look_up_lr: 'Look Up L/R', g2_lip_pucker: 'Lip Pucker', g2_eye_squint: 'Eye Squinting', g2_thinking_brow: 'Thinking Brow',
        g2_chin_stroke: 'Chin Stroke', g2_stillness: 'Stillness', g2_lowered_brow: 'Lowered Brow',
        g2_brow_furrow_deep: 'Brow Furrow Deep', g2_gaze_shift_frequency: 'Gaze Shift Frequency', g2_mouth_tight_eval: 'Mouth Tight (Eval)',
        g3_contempt: 'Contempt', g3_nose_crinkle: 'Nose Crinkle', g3_lip_compression: 'Lip Compression', g3_eye_block: 'Eye Block',
        g3_jaw_clench: 'Jaw Clenching', g3_rapid_blink: 'Rapid Blinking', g3_gaze_aversion: 'Gaze Aversion', g3_no_nod: 'No-Nod',
        g3_narrowed_pupils: 'Narrowed Pupils', g3_mouth_cover: 'Mouth Cover',
        g3_lip_corner_dip: 'Lip Corner Dip', g3_brow_lower_sustained: 'Brow Lower Sustained', g3_eye_squeeze: 'Eye Squeeze', g3_head_shake: 'Head Shake',
        g4_relaxed_exhale: 'Relaxed Exhale', g4_fixed_gaze: 'Fixed Gaze', g4_smile_transition: 'Smile to Genuine', g4_mouth_relax: 'Mouth Relax', g4_smile_sustain: 'Smile Sustain'
    };
    var SIGNIFIER_GROUPS = [
        { id: 'g1', title: 'Interest & Engagement', keys: ['g1_duchenne','g1_pupil_dilation','g1_eyebrow_flash','g1_eye_contact','g1_head_tilt','g1_forward_lean','g1_facial_symmetry','g1_rhythmic_nodding','g1_parted_lips','g1_softened_forehead','g1_micro_smile','g1_brow_raise_sustained','g1_mouth_open_receptive','g1_eye_widening','g1_nod_intensity'] },
        { id: 'g2', title: 'Cognitive Load', keys: ['g2_look_up_lr','g2_lip_pucker','g2_eye_squint','g2_thinking_brow','g2_chin_stroke','g2_stillness','g2_lowered_brow','g2_brow_furrow_deep','g2_gaze_shift_frequency','g2_mouth_tight_eval'] },
        { id: 'g3', title: 'Resistance & Objections', keys: ['g3_contempt','g3_nose_crinkle','g3_lip_compression','g3_eye_block','g3_jaw_clench','g3_rapid_blink','g3_gaze_aversion','g3_no_nod','g3_narrowed_pupils','g3_mouth_cover','g3_lip_corner_dip','g3_brow_lower_sustained','g3_eye_squeeze','g3_head_shake'] },
        { id: 'g4', title: 'Decision-Ready', keys: ['g4_relaxed_exhale','g4_fixed_gaze','g4_smile_transition','g4_mouth_relax','g4_smile_sustain'] }
    ];
    var COMPOSITE_LABELS = {
        verbal_nonverbal_alignment: 'Verbal–nonverbal alignment', cognitive_load_multimodal: 'Cognitive load (multimodal)',
        rapport_engagement: 'Rapport / engagement', skepticism_objection_strength: 'Skepticism / objection strength',
        decision_readiness_multimodal: 'Decision readiness (multimodal)', opportunity_strength: 'Opportunity strength',
        trust_rapport: 'Trust / rapport', disengagement_risk_multimodal: 'Disengagement risk (multimodal)',
        confusion_multimodal: 'Confusion (multimodal)', tension_objection_multimodal: 'Tension / objection (multimodal)',
        loss_of_interest_multimodal: 'Loss of interest (multimodal)', decision_plus_voice: 'Decision readiness + voice',
        psychological_safety_proxy: 'Psychological safety proxy', urgency_sensitivity: 'Urgency sensitivity',
        skepticism_strength: 'Skepticism strength', enthusiasm_multimodal: 'Enthusiasm (multimodal)',
        hesitation_multimodal: 'Hesitation (multimodal)', authority_deferral: 'Authority deferral',
        rapport_depth: 'Rapport depth', cognitive_overload_proxy: 'Cognitive overload proxy'
    };
    var COMPOSITE_KEYS = ['verbal_nonverbal_alignment','cognitive_load_multimodal','rapport_engagement','skepticism_objection_strength','decision_readiness_multimodal','opportunity_strength','trust_rapport','disengagement_risk_multimodal','confusion_multimodal','tension_objection_multimodal','loss_of_interest_multimodal','decision_plus_voice','psychological_safety_proxy','urgency_sensitivity','skepticism_strength','enthusiasm_multimodal','hesitation_multimodal','authority_deferral','rapport_depth','cognitive_overload_proxy'];
    var AZURE_BASE_LABELS = { anger: 'Anger', contempt: 'Contempt', disgust: 'Disgust', fear: 'Fear', happiness: 'Happiness', neutral: 'Neutral', sadness: 'Sadness', surprise: 'Surprise' };
    var AZURE_COMPOSITE_LABELS = { receptive: 'Receptive', focused: 'Focused', interested: 'Interested', agreeable: 'Agreeable', open: 'Open', skeptical: 'Skeptical', concerned: 'Concerned', disagreeing: 'Disagreeing', stressed: 'Stressed', disengaged: 'Disengaged' };

    function escapeHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

    function setBarAndValue(el, pct, text, color) {
        if (!el) return;
        var fill = el.querySelector && (el.querySelector('.metric-bar-fill') || el.querySelector('.signifier-bar-fill'));
        var val = el.querySelector && (el.querySelector('.metric-value') || el.querySelector('.signifier-value'));
        if (fill) { fill.style.width = (typeof pct === 'number' ? Math.max(0, Math.min(100, pct)) : 0) + '%'; if (color) fill.style.background = color; }
        if (val) val.textContent = text;
    }

    var engagementDOMBuilt = false;
    var BASIC_KEYS = ['attention','eyeContact','facialExpressiveness','headMovement','symmetry','mouthActivity'];
    var BASIC_LABELS = { attention: 'Attention', eyeContact: 'Eye contact', facialExpressiveness: 'Facial expressiveness', headMovement: 'Head movement', symmetry: 'Symmetry', mouthActivity: 'Mouth activity' };

    function buildEngagementDOMOnce() {
        if (engagementDOMBuilt) return;
        var basicEl = document.getElementById('engagementMetricsBasic');
        var sigEl = document.getElementById('engagementMetricsSignifiers');
        var compEl = document.getElementById('engagementMetricsComposite');
        if (basicEl) {
            var basicParts = ['<div class="metrics-section-title">Basic metrics (0–100)</div>'];
            for (var i = 0; i < BASIC_KEYS.length; i++) {
                var k = BASIC_KEYS[i];
                basicParts.push('<div class="metric-row" data-metric="' + k + '"><span class="metric-label">' + escapeHtml(BASIC_LABELS[k]) + '</span><div class="metric-bar-wrap"><div class="metric-bar-fill" style="width:0%;background:#64748b"></div></div><span class="metric-value">—</span></div>');
            }
            basicEl.innerHTML = basicParts.join('');
        }
        if (sigEl) {
            var sigParts = [];
            for (var g = 0; g < SIGNIFIER_GROUPS.length; g++) {
                var gr = SIGNIFIER_GROUPS[g];
                sigParts.push('<div class="metrics-section-title">' + escapeHtml(gr.title) + '</div><div class="signifier-list">');
                for (var j = 0; j < gr.keys.length; j++) {
                    var key = gr.keys[j];
                    sigParts.push('<div class="signifier-row" data-signifier="' + key + '"><span class="signifier-label">' + escapeHtml(SIGNIFIER_LABELS[key] || key) + '</span><div class="signifier-bar"><div class="signifier-bar-fill" style="width:0%;background:#64748b"></div></div><span class="signifier-value">—</span></div>');
                }
                sigParts.push('</div>');
            }
            sigEl.innerHTML = sigParts.length ? sigParts.join('') : '<div class="hint">No signifier data.</div>';
        }
        if (compEl) {
            var compParts = ['<div class="metrics-section-title">Composite (facial + speech + acoustics)</div><div class="signifier-list">'];
            for (var c = 0; c < COMPOSITE_KEYS.length; c++) {
                var key = COMPOSITE_KEYS[c];
                compParts.push('<div class="signifier-row" data-composite="' + key + '"><span class="signifier-label">' + escapeHtml(COMPOSITE_LABELS[key] || key) + '</span><div class="signifier-bar"><div class="signifier-bar-fill" style="width:0%;background:#64748b"></div></div><span class="signifier-value">—</span></div>');
            }
            compParts.push('</div>');
            compEl.innerHTML = compParts.join('');
        }
        engagementDOMBuilt = true;
    }

    function updateEngagementState(data) {
        var score = data.score != null ? Number(data.score) : NaN;
        var scorePct = (isNaN(score) || !isFinite(score)) ? 0 : Math.max(0, Math.min(100, score));
        var liveScoreEl = document.getElementById('engagementLiveScore');
        var liveLevelEl = document.getElementById('engagementLiveLevel');
        var liveBarEl = document.getElementById('engagementLiveBar');
        if (liveScoreEl) liveScoreEl.textContent = isNaN(score) || !isFinite(score) ? '—' : Math.round(score);
        if (liveLevelEl) liveLevelEl.textContent = data.level || '—';
        if (liveBarEl) {
            liveBarEl.style.width = scorePct + '%';
            liveBarEl.style.background = scorePct >= 70 ? '#22c55e' : scorePct >= 40 ? '#eab308' : '#64748b';
        }
        var summaryEl = document.getElementById('engagementMetricsSummary');
        if (summaryEl) {
            var parts = [
                'Face: ' + (data.faceDetected ? 'Yes' : 'No'),
                (data.detectionMethod ? data.detectionMethod : ''),
                (data.fps != null ? 'FPS: ' + data.fps.toFixed(1) : '')
            ].filter(Boolean);
            summaryEl.innerHTML = parts.length ? parts.map(function (p) { return '<span class="summary-pill">' + escapeHtml(p) + '</span>'; }).join('') : '';
        }
        buildEngagementDOMOnce();
        var m = data.metrics || {};
        for (var i = 0; i < BASIC_KEYS.length; i++) {
            var k = BASIC_KEYS[i];
            var v = m[k];
            var num = v != null ? Number(v) : NaN;
            var pct = (!isNaN(num) && isFinite(num)) ? Math.max(0, Math.min(100, num)) : 0;
            var row = document.querySelector('[data-metric="' + k + '"]');
            setBarAndValue(row, pct, (!isNaN(num) && isFinite(num)) ? Math.round(num).toString() : '—', barColor(pct));
        }
        var sigScores = data.signifierScores || {};
        for (var g = 0; g < SIGNIFIER_GROUPS.length; g++) {
            for (var j = 0; j < SIGNIFIER_GROUPS[g].keys.length; j++) {
                var key = SIGNIFIER_GROUPS[g].keys[j];
                var v = sigScores[key];
                var num = v != null ? Number(v) : NaN;
                var pct = (!isNaN(num) && isFinite(num)) ? Math.max(0, Math.min(100, num)) : 0;
                var row = document.querySelector('[data-signifier="' + key + '"]');
                setBarAndValue(row, pct, (!isNaN(num) && isFinite(num)) ? Math.round(num).toString() : '—', barColor(pct));
            }
        }
        var comp = data.compositeMetrics || {};
        for (var c = 0; c < COMPOSITE_KEYS.length; c++) {
            var key = COMPOSITE_KEYS[c];
            var v = comp[key];
            var num = v != null ? Number(v) : NaN;
            var pct = (!isNaN(num) && isFinite(num)) ? Math.max(0, Math.min(100, num)) : 0;
            var row = document.querySelector('[data-composite="' + key + '"]');
            setBarAndValue(row, pct, (!isNaN(num) && isFinite(num)) ? Math.round(num).toString() : '—', barColor(pct));
        }
        var azureEl = document.getElementById('engagementMetricsAzure');
        var azure = data.azureMetrics;
        if (azureEl) {
            if (azure && (azure.base || azure.composite)) {
                if (!document.querySelector('[data-azure-section]')) {
                    var azureHtml = '<div class="metrics-section-title" data-azure-section>Azure Face API</div>';
                    if (azure.score != null) azureHtml += '<div class="metric-row" data-azure="score"><span class="metric-label">Overall score</span><div class="metric-bar-wrap"><div class="metric-bar-fill" style="width:0%"></div></div><span class="metric-value">—</span></div>';
                    for (var bk in AZURE_BASE_LABELS) azureHtml += '<div class="metric-row" data-azure="base-' + bk + '"><span class="metric-label">' + escapeHtml(AZURE_BASE_LABELS[bk]) + '</span><div class="metric-bar-wrap"><div class="metric-bar-fill" style="width:0%"></div></div><span class="metric-value">—</span></div>';
                    azureHtml += '<div class="metrics-section-title" style="margin-top:12px">Azure B2B composites</div><div class="signifier-list">';
                    for (var ak in AZURE_COMPOSITE_LABELS) azureHtml += '<div class="signifier-row" data-azure="comp-' + ak + '"><span class="signifier-label">' + escapeHtml(AZURE_COMPOSITE_LABELS[ak]) + '</span><div class="signifier-bar"><div class="signifier-bar-fill" style="width:0%"></div></div><span class="signifier-value">—</span></div>';
                    azureHtml += '</div>';
                    azureEl.innerHTML = azureHtml;
                }
                var apct = azure.score != null ? Math.max(0, Math.min(100, azure.score)) : 0;
                setBarAndValue(document.querySelector('[data-azure="score"]'), apct, azure.score != null ? Math.round(azure.score).toString() : '—', barColor(apct));
                var baseObj = azure.base || {};
                for (var bk in AZURE_BASE_LABELS) { var bv = baseObj[bk]; var bn = bv != null ? Number(bv) : NaN; var bp = (!isNaN(bn) && isFinite(bn)) ? Math.max(0, Math.min(100, bn)) : 0; setBarAndValue(document.querySelector('[data-azure="base-' + bk + '"]'), bp, (!isNaN(bn) && isFinite(bn)) ? Math.round(bn).toString() : '—', barColor(bp)); }
                var compObj = azure.composite || {};
                for (var ak in AZURE_COMPOSITE_LABELS) { var av = compObj[ak]; var anum = av != null ? Number(av) : NaN; var ap = (!isNaN(anum) && isFinite(anum)) ? Math.max(0, Math.min(100, anum)) : 0; setBarAndValue(document.querySelector('[data-azure="comp-' + ak + '"]'), ap, (!isNaN(anum) && isFinite(anum)) ? Math.round(anum).toString() : '—', barColor(ap)); }
            } else {
                azureEl.innerHTML = '';
            }
        }
        var ctx = data.context || {};
        var ctxSummaryEl = document.getElementById('contextSummary');
        if (ctxSummaryEl) {
            var summaryText = (ctx.summary || '—').substring(0, 500) + (ctx.summary && ctx.summary.length > 500 ? '…' : '');
            ctxSummaryEl.textContent = summaryText;
            ctxSummaryEl.classList.toggle('empty', !ctx.summary || ctx.summary === '—');
        }
    }

    function setDashboardEngagementUI(running) {
        var liveEl = document.getElementById('dashboardEngagementStatus');
        var notEl = document.getElementById('dashboardEngagementNotRunning');
        if (liveEl) liveEl.style.display = running ? 'inline-flex' : 'none';
        if (notEl) notEl.style.display = running ? 'none' : 'inline';
    }

    function pollEngagement() {
        if (!document.getElementById('engagementCard')) return;
        var base = getApiBase();
        fetch(base + '/engagement/state?t=' + Date.now(), { method: 'GET', cache: 'no-cache' })
            .then(function (r) {
                if (!r.ok) {
                    setDashboardEngagementUI(false);
                    return null;
                }
                return r.json();
            })
            .then(function (data) {
                if (data) {
                    setDashboardEngagementUI(data.detectionStarted === true);
                    updateEngagementState(data);
                } else {
                    setDashboardEngagementUI(false);
                }
            })
            .catch(function () { setDashboardEngagementUI(false); });
    }

    function pollContext() {
        if (!document.getElementById('contextSent')) return;
        var base = getApiBase();
        fetch(base + '/api/engagement/context-and-response?t=' + Date.now(), { method: 'GET' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (!data) return;
                var ctxEl = document.getElementById('contextSent');
                var respEl = document.getElementById('lastResponse');
                var tsCtx = document.getElementById('contextTimestamp');
                var tsResp = document.getElementById('responseTimestamp');
                if (ctxEl) { var ctxText = (data.contextSent && data.contextSent.length) ? data.contextSent : null; ctxEl.textContent = ctxText || 'No context sent yet.'; ctxEl.classList.toggle('empty', !ctxText); }
                if (respEl) { var respText = (data.response && data.response.length) ? data.response : null; respEl.textContent = respText || 'No response yet.'; respEl.classList.toggle('empty', !respText); }
                if (data.timestamp) { var t = new Date(data.timestamp * 1000).toLocaleTimeString(); if (tsCtx) { tsCtx.textContent = 'Updated ' + t; tsCtx.classList.add('updated'); } if (tsResp) { tsResp.textContent = 'Updated ' + t; tsResp.classList.add('updated'); } }
            })
            .catch(function () {});
    }

    function bindButtons() {
        var appendBtn = document.getElementById('dashboardAppendContextBtn');
        var sendNowBtn = document.getElementById('dashboardSendNowBtn');
        var passStatus = document.getElementById('dashboardPassStatus');
        if (appendBtn) {
            appendBtn.addEventListener('click', function () {
                var input = document.getElementById('dashboardAdditionalContext');
                var text = (input && input.value) ? input.value.trim() : '';
                appendBtn.disabled = true;
                if (passStatus) passStatus.textContent = '';
                fetch(getApiBase() + '/api/engagement/set-additional-context', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ additionalContext: text || null })
                })
                    .then(function (r) {
                        if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || d.details || 'Request failed'); }).catch(function () { throw new Error('Request failed'); });
                        return r.json();
                    })
                    .then(function () {
                        if (passStatus) {
                            passStatus.textContent = text ? 'Saved. Will be included in the next context sent to Foundry.' : 'Cleared.';
                            passStatus.className = 'dashboard-pass-status success';
                        }
                        setTimeout(function () { if (passStatus) { passStatus.textContent = ''; passStatus.className = 'dashboard-pass-status'; } }, 4000);
                    })
                    .catch(function (err) {
                        if (passStatus) { passStatus.textContent = err.message || 'Failed'; passStatus.className = 'dashboard-pass-status error'; }
                        setTimeout(function () { if (passStatus) passStatus.className = 'dashboard-pass-status'; }, 4000);
                    })
                    .finally(function () { appendBtn.disabled = false; });
            });
        }
        if (sendNowBtn) {
            sendNowBtn.addEventListener('click', function () {
                sendNowBtn.disabled = true;
                if (passStatus) passStatus.textContent = '';
                fetch(getApiBase() + '/api/context-push', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
                    .then(function (r) {
                        if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || d.details || 'Request failed'); }).catch(function () { throw new Error('Request failed'); });
                        return r.json();
                    })
                    .then(function (data) {
                        if (data.context != null) {
                            var ctxEl = document.getElementById('contextSent');
                            if (ctxEl) { ctxEl.textContent = data.context; ctxEl.classList.remove('empty'); }
                        }
                        if (data.response != null) {
                            var respEl = document.getElementById('lastResponse');
                            if (respEl) { respEl.textContent = data.response; respEl.classList.remove('empty'); }
                        }
                        var tsCtx = document.getElementById('contextTimestamp');
                        var tsResp = document.getElementById('responseTimestamp');
                        var t = new Date().toLocaleTimeString();
                        if (tsCtx) { tsCtx.textContent = 'Updated ' + t; tsCtx.classList.add('updated'); }
                        if (tsResp) { tsResp.textContent = 'Updated ' + t; tsResp.classList.add('updated'); }
                        if (passStatus) { passStatus.textContent = 'Updated'; passStatus.className = 'dashboard-pass-status success'; }
                        setTimeout(function () { if (passStatus) { passStatus.textContent = ''; passStatus.className = 'dashboard-pass-status'; } }, 3000);
                    })
                    .catch(function (err) {
                        if (passStatus) { passStatus.textContent = err.message || 'Failed'; passStatus.className = 'dashboard-pass-status error'; }
                        setTimeout(function () { if (passStatus) passStatus.className = 'dashboard-pass-status'; }, 4000);
                    })
                    .finally(function () { sendNowBtn.disabled = false; });
            });
        }
    }

    if (!document.getElementById('engagementCard')) return;

    setDashboardEngagementUI(false);
    pollEngagement();
    pollContext();
    setInterval(pollEngagement, 500);
    setInterval(pollContext, 2500);
    bindButtons();
})();
