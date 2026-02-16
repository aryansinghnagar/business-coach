// ========================================================================
// ACOUSTIC ANALYZER (acoustic-analyzer.js)
// ========================================================================
//
// WHAT THIS MODULE DOES (in plain language):
// ------------------------------------------
// Analyzes the meeting partner (or mic) audio in the browser. For each short
// window it computes: loudness, pitch, pitch contour (rising/falling/flat),
// variability, and a simple tone proxy. When speech is active it POSTs these
// windows to the backend (POST /engagement/acoustic-context) so the server can
// summarize voice cues and feed them into engagement and AI context.
//
// WHO USES THIS: app-main.js starts and stops the analyzer when the insights
// session starts or ends; the backend uses the data for context and composites.
//

(function () {
  "use strict";
  var WINDOW_MS = 1500;
  var SAMPLE_INTERVAL_MS = 100;
  var SPEECH_THRESHOLD = 0.08;
  var MIN_PITCH_HZ = 80;
  var MAX_PITCH_HZ = 400;
  var PITCH_CONTOUR_DELTA_SEMITONES = 1.5;
  var VOICING_CONFIDENCE_THRESHOLD = 0.5;
  var CREPE_MODEL_URL = 'https://cdn.jsdelivr.net/gh/ml5js/ml5-data-and-models@1/models/pitch-detection/crepe/';

  var _context = null;
  var _source = null;
  var _analyser = null;
  var _scriptNode = null;
  var _intervalId = null;
  var _crepePollId = null;
  var _apiBase = '';
  var _enabled = true;
  var _pitchHistory = [];
  var _lastPitchHz = null;
  var _crepePitch = null;
  var _crepeReady = false;
  var _crepePitchHz = null;
  var _crepeVoicingConfidence = null;

  function hzToSemitones(hz) {
    if (!hz || hz <= 0) return 0;
    return 12 * Math.log(hz / 261.626) / Math.LN2;
  }

  function semitonesDelta(hz1, hz2) {
    if (!hz1 || !hz2 || hz1 <= 0 || hz2 <= 0) return 0;
    return hzToSemitones(hz2) - hzToSemitones(hz1);
  }

  function pitchVariabilitySemitones(pitchHistory) {
    if (!pitchHistory || pitchHistory.length < 2) return 0;
    var semis = pitchHistory.map(function (h) { return hzToSemitones(h); });
    var mean = semis.reduce(function (a, b) { return a + b; }, 0) / semis.length;
    var variance = semis.reduce(function (acc, s) { return acc + (s - mean) * (s - mean); }, 0) / semis.length;
    return Math.sqrt(variance);
  }

  function getPitchContour(prevHz, currHz) {
    if (prevHz == null || currHz == null || prevHz <= 0 || currHz <= 0) return 'flat';
    var delta = semitonesDelta(prevHz, currHz);
    if (delta >= PITCH_CONTOUR_DELTA_SEMITONES) return 'rising';
    if (delta <= -PITCH_CONTOUR_DELTA_SEMITONES) return 'falling';
    return 'flat';
  }

  function initCrepe(stream, audioContext) {
    if (typeof ml5 === 'undefined' || !ml5.pitchDetection) return;
    try {
      ml5.pitchDetection(CREPE_MODEL_URL, audioContext, stream, function (err) {
        if (err) {
          console.warn('CREPE pitch model failed to load:', err);
          return;
        }
        _crepePitch = arguments[0];
        _crepeReady = true;
        startCrepePoll();
      });
    } catch (e) {
      console.warn('CREPE pitch init failed:', e);
    }
  }

  function startCrepePoll() {
    stopCrepePoll();
    function poll() {
      if (!_crepePitch || !_crepeReady) {
        _crepePollId = setTimeout(poll, SAMPLE_INTERVAL_MS);
        return;
      }
      _crepePitch.getPitch(function (err, freq) {
        var conf = null;
        if (_crepePitch.results && _crepePitch.results.confidence != null) {
          conf = parseFloat(_crepePitch.results.confidence);
        }
        if (!err && freq != null && freq >= MIN_PITCH_HZ && freq <= MAX_PITCH_HZ) {
          _crepePitchHz = freq;
          _crepeVoicingConfidence = conf != null ? conf : 1;
        } else {
          _crepePitchHz = null;
          _crepeVoicingConfidence = conf != null ? conf : 0;
        }
      });
      _crepePollId = setTimeout(poll, SAMPLE_INTERVAL_MS);
    }
    poll();
  }

  function stopCrepePoll() {
    if (_crepePollId) {
      clearTimeout(_crepePollId);
      _crepePollId = null;
    }
    _crepePitchHz = null;
    _crepeVoicingConfidence = null;
  }

  /**
   * Start analyzing stream and posting windowed features to the backend.
   * @param {MediaStream} stream - Same stream as STT (partner or mic)
   * @param {string} apiBaseUrl - Base URL for API (e.g. http://localhost:5000)
   * @param {boolean} enabled - If false, no POSTs (feature flag from config)
   */
  function start(stream, apiBaseUrl, enabled) {
    stop();
    if (!stream || stream.getAudioTracks().length === 0) return;
    _enabled = enabled !== false;
    _apiBase = (apiBaseUrl || '').replace(/\/$/, '');
    _pitchHistory = [];
    _lastPitchHz = null;
    _crepePitch = null;
    _crepeReady = false;
    _crepePitchHz = null;
    _crepeVoicingConfidence = null;

    try {
      _context = new (window.AudioContext || window.webkitAudioContext)();
      _analyser = _context.createAnalyser();
      _analyser.fftSize = 2048;
      _analyser.smoothingTimeConstant = 0.6;
      _analyser.minDecibels = -60;
      _analyser.maxDecibels = -10;
      _source = _context.createMediaStreamSource(stream);
      _source.connect(_analyser);

      initCrepe(stream, _context);

      var bufferLength = _analyser.frequencyBinCount;
      var dataArray = new Uint8Array(bufferLength);
      var timeData = new Uint8Array(_analyser.fftSize);
      var sampleRate = _context.sampleRate;

      var windowSamples = [];
      var windowStart = Date.now();

      function tick() {
        if (!_analyser) return;
        _analyser.getByteTimeDomainData(timeData);
        var sumSq = 0;
        for (var i = 0; i < timeData.length; i++) {
          var n = (timeData[i] - 128) / 128;
          sumSq += n * n;
        }
        var rms = Math.sqrt(sumSq / timeData.length);
        _analyser.getByteFrequencyData(dataArray);
        var lowStart = Math.floor((MIN_PITCH_HZ * bufferLength * 2) / sampleRate);
        var lowEnd = Math.ceil((MAX_PITCH_HZ * bufferLength * 2) / sampleRate);
        lowStart = Math.max(0, lowStart);
        lowEnd = Math.min(bufferLength, lowEnd);
        var maxMag = 0;
        var maxBin = lowStart;
        for (var j = lowStart; j < lowEnd; j++) {
          if (dataArray[j] > maxMag) {
            maxMag = dataArray[j];
            maxBin = j;
          }
        }
        var fftPitchHz = maxMag > 2 ? (maxBin * sampleRate) / (_analyser.fftSize) : null;
        if (fftPitchHz && (fftPitchHz < MIN_PITCH_HZ || fftPitchHz > MAX_PITCH_HZ)) fftPitchHz = null;

        var pitchHz = null;
        var voicingConfidence = null;
        if (_crepeReady && _crepePitchHz != null && _crepePitchHz >= MIN_PITCH_HZ && _crepePitchHz <= MAX_PITCH_HZ) {
          pitchHz = _crepePitchHz;
          voicingConfidence = _crepeVoicingConfidence != null ? Math.min(1, Math.max(0, _crepeVoicingConfidence)) : 1;
        }
        if (pitchHz == null) {
          pitchHz = fftPitchHz;
          voicingConfidence = fftPitchHz ? 0.5 : null;
        }

        var pitchForContourVar = (voicingConfidence != null && voicingConfidence < VOICING_CONFIDENCE_THRESHOLD)
          ? null : pitchHz;
        if (pitchForContourVar && pitchForContourVar >= MIN_PITCH_HZ && pitchForContourVar <= MAX_PITCH_HZ) {
          _pitchHistory.push(pitchForContourVar);
          if (_pitchHistory.length > 10) _pitchHistory.shift();
        }
        var contour = getPitchContour(_lastPitchHz, pitchForContourVar);
        if (pitchForContourVar) _lastPitchHz = pitchForContourVar;

        var spectralSum = 0;
        var magnitudeSum = 0;
        for (var k = 0; k < bufferLength; k++) {
          var mag = dataArray[k] / 255;
          magnitudeSum += mag;
          spectralSum += k * mag;
        }
        var centroidBin = magnitudeSum > 0 ? spectralSum / magnitudeSum : 0;
        var toneProxy = magnitudeSum > 0 ? Math.min(1, centroidBin / bufferLength) : 0;

        windowSamples.push({
          rms: rms,
          pitchHz: pitchHz,
          voicingConfidence: voicingConfidence,
          pitchForContourVar: pitchForContourVar,
          contour: contour,
          toneProxy: toneProxy
        });

        var elapsed = Date.now() - windowStart;
        if (elapsed >= WINDOW_MS) {
          var avgRms = windowSamples.reduce(function (a, s) { return a + s.rms; }, 0) / windowSamples.length;
          var speechActive = avgRms >= SPEECH_THRESHOLD;
          if (speechActive && _enabled) {
            var avgPitch = null;
            var pitchCount = 0;
            var avgConfidence = null;
            var confCount = 0;
            windowSamples.forEach(function (s) {
              if (s.pitchHz) { avgPitch = (avgPitch || 0) + s.pitchHz; pitchCount++; }
              if (s.voicingConfidence != null) { avgConfidence = (avgConfidence || 0) + s.voicingConfidence; confCount++; }
            });
            avgPitch = pitchCount > 0 ? avgPitch / pitchCount : null;
            avgConfidence = confCount > 0 ? avgConfidence / confCount : null;
            var variability = pitchVariabilitySemitones(_pitchHistory);
            var lastContour = windowSamples.length ? windowSamples[windowSamples.length - 1].contour : 'flat';
            var avgTone = windowSamples.reduce(function (a, s) { return a + s.toneProxy; }, 0) / windowSamples.length;
            var payload = {
              windows: [{
                loudness_norm: Math.min(1, avgRms * 4),
                pitch_hz: avgPitch,
                pitch_contour: lastContour,
                pitch_variability: variability,
                tone_proxy: avgTone,
                voicing_confidence: avgConfidence,
                speech_active: true
              }]
            };
            var url = _apiBase ? _apiBase + '/engagement/acoustic-context' : '/engagement/acoustic-context';
            fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            }).catch(function (err) { console.warn('Acoustic context POST failed:', err); });
          }
          windowSamples.length = 0;
          windowStart = Date.now();
        }
      }

      _intervalId = setInterval(tick, SAMPLE_INTERVAL_MS);
    } catch (err) {
      console.warn('Acoustic analyzer start failed:', err);
      stop();
    }
  }

  /**
   * Stop analysis and release resources.
   */
  function stop() {
    if (_intervalId) {
      clearInterval(_intervalId);
      _intervalId = null;
    }
    stopCrepePoll();
    _crepePitch = null;
    _crepeReady = false;
    _crepePitchHz = null;
    _crepeVoicingConfidence = null;
    if (_scriptNode && _context) {
      try { _scriptNode.disconnect(); } catch (e) {}
      _scriptNode = null;
    }
    if (_source && _context) {
      try { _source.disconnect(); } catch (e) {}
      _source = null;
    }
    _analyser = null;
    if (_context && _context.state !== 'closed') {
      _context.close().catch(function () {});
    }
    _context = null;
    _pitchHistory = [];
    _lastPitchHz = null;
  }

  window.AcousticAnalyzer = { start: start, stop: stop };
})();
