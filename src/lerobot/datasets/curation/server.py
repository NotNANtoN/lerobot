# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive Web-based Curation & Observability Server for LeRobot Datasets.

Zero external front-end dependencies: runs directly via Python standard library HTTP server,
providing video playback, synchronized trajectory charts, anomaly breakdown, interactive trimming,
and one-click clean dataset / split export.
"""

from __future__ import annotations

import json
import logging
import re
import socketserver
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

from lerobot.datasets.curation.curator import DatasetCurator
from lerobot.datasets.curation.manifest import CurationManifest

logger = logging.getLogger(__name__)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LeRobot Dataset Observability & Curation</title>
  <style>
    :root {
      --bg-dark: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-dim: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --keep-color: #22c55e;
      --drop-color: #ef4444;
      --review-color: #f59e0b;
      --highlight: #6366f1;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #111827;
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    .header-title {
      font-size: 1.25rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 9999px;
      font-weight: 600;
    }
    .badge-repo { background: #3b82f6; color: white; }
    .badge-keep { background: #15803d; color: #dcfce7; }
    .badge-drop { background: #b91c1c; color: #fee2e2; }
    .badge-review { background: #b45309; color: #fef3c7; }

    .header-actions {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    button {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.875rem;
      transition: all 0.2s;
    }
    button:hover { background: var(--accent-hover); color: white; }
    button.secondary { background: #334155; color: var(--text-main); }
    button.secondary:hover { background: #475569; }
    button.btn-keep { background: var(--keep-color); color: white; }
    button.btn-drop { background: var(--drop-color); color: white; }
    button.btn-review { background: var(--review-color); color: white; }

    .main-container {
      display: flex;
      flex: 1;
      height: calc(100vh - 65px);
      overflow: hidden;
    }

    /* Left Sidebar: Episode List */
    .sidebar {
      width: 340px;
      background: #111827;
      border-right: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    .sidebar-filter {
      padding: 12px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .filter-tabs {
      display: flex;
      gap: 4px;
    }
    .filter-btn {
      flex: 1;
      font-size: 0.75rem;
      padding: 4px 6px;
      background: #1e293b;
      color: var(--text-dim);
      border-radius: 4px;
      border: 1px solid transparent;
      cursor: pointer;
    }
    .filter-btn.active {
      background: #334155;
      color: var(--text-main);
      border-color: var(--accent);
    }
    .episodes-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .episode-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 10px 12px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 6px;
      transition: border-color 0.15s;
    }
    .episode-card:hover { border-color: var(--accent); }
    .episode-card.active {
      border-color: var(--accent);
      background: #1e3a8a33;
    }
    .ep-card-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .ep-title { font-weight: 700; font-size: 0.95rem; }
    .ep-score {
      font-size: 0.8rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
    }
    .score-high { background: #ef444433; color: #f87171; border: 1px solid #ef4444; }
    .score-mid { background: #f59e0b33; color: #fbbf24; border: 1px solid #f59e0b; }
    .score-low { background: #22c55e33; color: #4ade80; border: 1px solid #22c55e; }

    .ep-tags {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
    }
    .tag-chip {
      font-size: 0.7rem;
      background: #334155;
      color: var(--text-dim);
      padding: 2px 6px;
      border-radius: 4px;
    }

    /* Content Area */
    .content-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 20px;
      gap: 20px;
    }

    .top-panel {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 20px;
    }

    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .video-container {
      position: relative;
      width: 100%;
      background: black;
      border-radius: 6px;
      overflow: hidden;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .trim-controls {
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: #0f172a;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--card-border);
    }
    .trim-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
    }
    .range-inputs {
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
      margin: 0 12px;
    }
    input[type="range"] {
      flex: 1;
      cursor: pointer;
    }
    input[type="number"] {
      width: 70px;
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 4px 6px;
      border-radius: 4px;
      text-align: center;
    }

    .episode-details-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .status-actions {
      display: flex;
      gap: 10px;
    }
    .status-actions button {
      flex: 1;
      padding: 10px;
      font-size: 0.9rem;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .metric-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      padding: 10px;
      border-radius: 6px;
    }
    .metric-title { font-size: 0.75rem; color: var(--text-dim); }
    .metric-val { font-size: 1.1rem; font-weight: 700; color: var(--accent); margin-top: 2px; }

    .bottom-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    #trajectory-canvas {
      width: 100%;
      height: 220px;
      background: #0f172a;
      border-radius: 6px;
      border: 1px solid var(--card-border);
    }

    .notes-input {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 8px;
      border-radius: 6px;
      font-size: 0.85rem;
      resize: vertical;
    }

    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #059669;
      color: white;
      padding: 10px 18px;
      border-radius: 6px;
      font-weight: 600;
      opacity: 0;
      transition: opacity 0.3s;
      pointer-events: none;
      z-index: 1000;
    }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <span>Robot Dataset Observability</span>
      <span class="badge badge-repo" id="repo-id-badge">Loading...</span>
      <span class="badge badge-keep" id="kept-count">Kept: 0</span>
      <span class="badge badge-review" id="review-count">Review: 0</span>
      <span class="badge badge-drop" id="drop-count">Drop: 0</span>
    </div>
    <div class="header-actions">
      <button class="secondary" onclick="saveManifest()">💾 Save Manifest</button>
      <button class="secondary" onclick="exportCleanSplit()">Export Split JSON</button>
      <button onclick="exportMaterializedDataset()">Export Clean Dataset</button>
    </div>
  </header>

  <div class="main-container">
    <!-- Left Sidebar: Episodes list -->
    <div class="sidebar">
      <div class="sidebar-filter">
        <div class="filter-tabs">
          <button class="filter-btn active" onclick="setFilter('all')">All</button>
          <button class="filter-btn" onclick="setFilter('outliers')">Outliers</button>
          <button class="filter-btn" onclick="setFilter('review')">Review</button>
          <button class="filter-btn" onclick="setFilter('keep')">Keep</button>
          <button class="filter-btn" onclick="setFilter('drop')">Drop</button>
        </div>
      </div>
      <div class="episodes-list" id="episodes-container">
        <!-- Rendered items -->
      </div>
    </div>

    <!-- Main Content Area -->
    <div class="content-area">
      <div class="top-panel">
        <!-- Video & Scrubbing -->
        <div class="video-card">
          <div class="ep-card-row">
            <h3 id="current-ep-title">Episode 0</h3>
            <span id="current-frame-indicator" style="font-size: 0.85rem; color: var(--accent);">Frame: 0 / 0</span>
          </div>
          <div class="video-container">
            <video id="episode-video" controls muted></video>
          </div>
          <div class="trim-controls">
            <div class="trim-row">
              <strong>Interactive Trimming:</strong>
              <button class="secondary" style="padding: 4px 8px; font-size: 0.75rem;" onclick="autoTrimCurrent()">✨ Auto-Trim Motion</button>
            </div>
            <div class="trim-row">
              <span>Start Frame:</span>
              <div class="range-inputs">
                <input type="range" id="trim-start-range" min="0" max="100" value="0" oninput="onTrimSliderChange()">
                <input type="number" id="trim-start-input" min="0" max="100" value="0" onchange="onTrimInputChange()">
              </div>
            </div>
            <div class="trim-row">
              <span>End Frame:</span>
              <div class="range-inputs">
                <input type="range" id="trim-end-range" min="0" max="100" value="100" oninput="onTrimSliderChange()">
                <input type="number" id="trim-end-input" min="0" max="100" value="100" onchange="onTrimInputChange()">
              </div>
            </div>
          </div>
        </div>

        <!-- Anomaly breakdown & status actions -->
        <div class="episode-details-card">
          <div class="ep-card-row">
            <h3>Curation Decision</h3>
            <span class="badge" id="current-status-badge">KEEP</span>
          </div>

          <div class="status-actions">
            <button class="btn-keep" onclick="setEpisodeStatus('keep')">Keep [K]</button>
            <button class="btn-review" onclick="setEpisodeStatus('review')">Review [R]</button>
            <button class="btn-drop" onclick="setEpisodeStatus('drop')">Drop [D]</button>
          </div>

          <div class="metrics-grid">
            <div class="metric-box">
              <div class="metric-title">Anomaly Score</div>
              <div class="metric-val" id="metric-anomaly">0.0</div>
            </div>
            <div class="metric-box">
              <div class="metric-title">DTW Distance (Proprio)</div>
              <div class="metric-val" id="metric-dtw">0.0</div>
            </div>
            <div class="metric-box">
              <div class="metric-title">Jerk / Smoothness</div>
              <div class="metric-val" id="metric-smoothness">0.0</div>
            </div>
            <div class="metric-box">
              <div class="metric-title">Duration / Frames</div>
              <div class="metric-val" id="metric-duration">0s (0)</div>
            </div>
          </div>

          <div>
            <div class="metric-title" style="margin-bottom: 6px;">Anomaly Findings:</div>
            <div id="reasons-container" style="display: flex; flex-direction: column; gap: 4px;">
              <span class="tag-chip">Nominal demonstration</span>
            </div>
          </div>

          <div>
            <div class="metric-title" style="margin-bottom: 6px;">Curator Notes:</div>
            <textarea class="notes-input" id="episode-notes" rows="2" placeholder="Add notes..." onchange="onNotesChange()"></textarea>
          </div>
        </div>
      </div>

      <!-- Synchronized Trajectory Canvas -->
      <div class="bottom-panel">
        <div class="chart-header">
          <h3>Proprioceptive Trajectory Curves (Joint Positions & Gripper)</h3>
          <div style="font-size: 0.8rem; color: var(--text-dim);">
            Green: Trim Start | Red: Trim End | White: Playhead
          </div>
        </div>
        <canvas id="trajectory-canvas" width="1200" height="220"></canvas>
      </div>
    </div>
  </div>

  <div class="toast" id="toast-msg">Saved Successfully!</div>

  <script>
    let manifestData = null;
    let currentEpisodeIndex = 0;
    let currentEpisodeData = null;
    let activeFilter = 'all';

    const colors = ['#38bdf8', '#fb7185', '#a78bfa', '#facc15', '#34d399', '#f97316'];

    async function init() {
      const res = await fetch('/api/manifest');
      manifestData = await res.json();
      updateHeader();
      renderEpisodeList();
      if (manifestData.episodes && Object.keys(manifestData.episodes).length > 0) {
        loadEpisode(parseInt(Object.keys(manifestData.episodes)[0]));
      }
      setupKeyboardShortcuts();
    }

    function updateHeader() {
      document.getElementById('repo-id-badge').innerText = manifestData.repo_id;
      const s = manifestData.summary;
      document.getElementById('kept-count').innerText = `Kept: ${s.kept_episodes || 0}`;
      document.getElementById('review-count').innerText = `Review: ${s.review_episodes || 0}`;
      document.getElementById('drop-count').innerText = `Drop: ${s.dropped_episodes || 0}`;
    }

    function setFilter(f) {
      activeFilter = f;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      renderEpisodeList();
    }

    function renderEpisodeList() {
      const container = document.getElementById('episodes-container');
      container.innerHTML = '';
      const eps = Object.values(manifestData.episodes);
      eps.sort((a, b) => b.anomaly_score - a.anomaly_score);

      for (const ep of eps) {
        if (activeFilter === 'outliers' && !ep.tags.includes('outlier')) continue;
        if (activeFilter === 'review' && ep.status !== 'review') continue;
        if (activeFilter === 'keep' && ep.status !== 'keep') continue;
        if (activeFilter === 'drop' && ep.status !== 'drop') continue;

        const card = document.createElement('div');
        card.className = `episode-card ${ep.episode_index === currentEpisodeIndex ? 'active' : ''}`;
        card.onclick = () => loadEpisode(ep.episode_index);

        let scoreClass = 'score-low';
        if (ep.anomaly_score > 60) scoreClass = 'score-high';
        else if (ep.anomaly_score > 30) scoreClass = 'score-mid';

        card.innerHTML = `
          <div class="ep-card-row">
            <span class="ep-title">Ep ${ep.episode_index}</span>
            <span class="ep-score ${scoreClass}">${ep.anomaly_score.toFixed(1)}</span>
          </div>
          <div class="ep-card-row" style="font-size: 0.8rem; color: var(--text-dim);">
            <span>${ep.trimmed_length} frames</span>
            <span class="badge badge-${ep.status}">${ep.status.toUpperCase()}</span>
          </div>
          <div class="ep-tags">
            ${ep.tags.map(t => `<span class="tag-chip">${t}</span>`).join('')}
          </div>
        `;
        container.appendChild(card);
      }
    }

    async function loadEpisode(epIdx) {
      currentEpisodeIndex = epIdx;
      renderEpisodeList();

      const ep = manifestData.episodes[epIdx.toString()];
      document.getElementById('current-ep-title').innerText = `Episode ${epIdx}`;
      document.getElementById('episode-notes').value = ep.notes || '';

      const badge = document.getElementById('current-status-badge');
      badge.className = `badge badge-${ep.status}`;
      badge.innerText = ep.status.toUpperCase();

      document.getElementById('metric-anomaly').innerText = ep.anomaly_score.toFixed(1);
      document.getElementById('metric-dtw').innerText = ep.proprio_anomaly_score.toFixed(3);
      document.getElementById('metric-smoothness').innerText = (ep.kinematics.smoothness_score || 0).toFixed(2);
      document.getElementById('metric-duration').innerText = `${(ep.kinematics.duration_seconds || 0).toFixed(1)}s (${ep.original_length})`;

      const reasonsBox = document.getElementById('reasons-container');
      reasonsBox.innerHTML = '';
      if (ep.anomaly_reasons && ep.anomaly_reasons.length > 0) {
        for (const r of ep.anomaly_reasons) {
          const chip = document.createElement('span');
          chip.className = 'tag-chip';
          chip.style.color = '#f87171';
          chip.innerText = '⚠ ' + r;
          reasonsBox.appendChild(chip);
        }
      } else {
        reasonsBox.innerHTML = '<span class="tag-chip">Nominal demonstration</span>';
      }

      // Sliders & inputs
      const maxFrames = ep.original_length;
      const sRange = document.getElementById('trim-start-range');
      const sInput = document.getElementById('trim-start-input');
      const eRange = document.getElementById('trim-end-range');
      const eInput = document.getElementById('trim-end-input');

      sRange.max = maxFrames;
      sInput.max = maxFrames;
      eRange.max = maxFrames;
      eInput.max = maxFrames;

      sRange.value = ep.trim_start;
      sInput.value = ep.trim_start;
      eRange.value = ep.trim_end;
      eInput.value = ep.trim_end;

      // Video load
      const video = document.getElementById('episode-video');
      video.src = `/api/video/${epIdx}`;
      video.ontimeupdate = () => {
        const frame = Math.floor(video.currentTime * manifestData.fps);
        document.getElementById('current-frame-indicator').innerText = `Frame: ${frame} / ${maxFrames}`;
        drawTrajectoryCanvas(frame);
      };

      // Load trajectory curves
      const res = await fetch(`/api/episode/${epIdx}`);
      currentEpisodeData = await res.json();
      drawTrajectoryCanvas(0);
    }

    function onTrimSliderChange() {
      const s = parseInt(document.getElementById('trim-start-range').value);
      const e = parseInt(document.getElementById('trim-end-range').value);
      document.getElementById('trim-start-input').value = s;
      document.getElementById('trim-end-input').value = e;
      updateTrimBounds(s, e);
    }

    function onTrimInputChange() {
      const s = parseInt(document.getElementById('trim-start-input').value);
      const e = parseInt(document.getElementById('trim-end-input').value);
      document.getElementById('trim-start-range').value = s;
      document.getElementById('trim-end-range').value = e;
      updateTrimBounds(s, e);
    }

    function updateTrimBounds(start, end) {
      const ep = manifestData.episodes[currentEpisodeIndex.toString()];
      ep.trim_start = Math.max(0, Math.min(start, end));
      ep.trim_end = Math.min(ep.original_length, Math.max(start, end));
      drawTrajectoryCanvas(Math.floor(document.getElementById('episode-video').currentTime * manifestData.fps));
      syncEpisodeUpdate();
    }

    function autoTrimCurrent() {
      const ep = manifestData.episodes[currentEpisodeIndex.toString()];
      if (ep.kinematics) {
        ep.trim_start = ep.kinematics.suggested_trim_start || 0;
        ep.trim_end = ep.kinematics.suggested_trim_end || ep.original_length;
        document.getElementById('trim-start-range').value = ep.trim_start;
        document.getElementById('trim-start-input').value = ep.trim_start;
        document.getElementById('trim-end-range').value = ep.trim_end;
        document.getElementById('trim-end-input').value = ep.trim_end;
        drawTrajectoryCanvas(0);
        syncEpisodeUpdate();
        showToast('Auto-trimmed to detected motion!');
      }
    }

    function setEpisodeStatus(status) {
      const ep = manifestData.episodes[currentEpisodeIndex.toString()];
      ep.status = status;
      const badge = document.getElementById('current-status-badge');
      badge.className = `badge badge-${status}`;
      badge.innerText = status.toUpperCase();
      renderEpisodeList();
      syncEpisodeUpdate();
      showToast(`Marked Episode ${currentEpisodeIndex} as ${status.toUpperCase()}`);
    }

    function onNotesChange() {
      const ep = manifestData.episodes[currentEpisodeIndex.toString()];
      ep.notes = document.getElementById('episode-notes').value;
      syncEpisodeUpdate();
    }

    async function syncEpisodeUpdate() {
      const ep = manifestData.episodes[currentEpisodeIndex.toString()];
      await fetch('/api/update_episode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ep)
      });
      updateHeader();
    }

    function drawTrajectoryCanvas(playheadFrame) {
      if (!currentEpisodeData || !currentEpisodeData.states) return;
      const canvas = document.getElementById('trajectory-canvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      const states = currentEpisodeData.states;
      const T = states.length;
      if (T === 0) return;
      const D = states[0].length;

      // Find min / max across all dimensions
      let minVal = Infinity, maxVal = -Infinity;
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) {
          if (states[t][d] < minVal) minVal = states[t][d];
          if (states[t][d] > maxVal) maxVal = states[t][d];
        }
      }
      const range = (maxVal - minVal) || 1.0;

      // Draw grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let y = 20; y < h; y += 40) {
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
      }
      ctx.stroke();

      // Draw trim zones (grayed out regions)
      const ep = manifestData.episodes[currentEpisodeIndex.toString()];
      const trimStartX = (ep.trim_start / T) * w;
      const trimEndX = (ep.trim_end / T) * w;

      ctx.fillStyle = '#0f172a88';
      ctx.fillRect(0, 0, trimStartX, h);
      ctx.fillRect(trimEndX, 0, w - trimEndX, h);

      // Draw trim boundary lines
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(trimStartX, 0); ctx.lineTo(trimStartX, h);
      ctx.stroke();

      ctx.strokeStyle = '#ef4444';
      ctx.beginPath();
      ctx.moveTo(trimEndX, 0); ctx.lineTo(trimEndX, h);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw joint curves
      ctx.lineWidth = 2;
      for (let d = 0; d < D; d++) {
        ctx.strokeStyle = colors[d % colors.length];
        ctx.beginPath();
        for (let t = 0; t < T; t++) {
          const x = (t / (T - 1)) * w;
          const normY = (states[t][d] - minVal) / range;
          const y = h - 20 - (normY * (h - 40));
          if (t === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Draw playhead
      if (playheadFrame !== undefined && playheadFrame >= 0) {
        const playX = (playheadFrame / T) * w;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playX, 0);
        ctx.lineTo(playX, h);
        ctx.stroke();
      }
    }

    function setupKeyboardShortcuts() {
      window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'k' || e.key === 'K') setEpisodeStatus('keep');
        if (e.key === 'd' || e.key === 'D') setEpisodeStatus('drop');
        if (e.key === 'r' || e.key === 'R') setEpisodeStatus('review');
        if (e.key === 'ArrowRight') {
          const keys = Object.keys(manifestData.episodes).map(Number);
          const next = keys[(keys.indexOf(currentEpisodeIndex) + 1) % keys.length];
          loadEpisode(next);
        }
        if (e.key === 'ArrowLeft') {
          const keys = Object.keys(manifestData.episodes).map(Number);
          const prev = keys[(keys.indexOf(currentEpisodeIndex) - 1 + keys.length) % keys.length];
          loadEpisode(prev);
        }
        if (e.key === ' ') {
          const v = document.getElementById('episode-video');
          if (v.paused) v.play(); else v.pause();
          e.preventDefault();
        }
      });
    }

    async function saveManifest() {
      const res = await fetch('/api/save_manifest', { method: 'POST' });
      const data = await res.json();
      showToast('Manifest saved to ' + data.path);
    }

    async function exportCleanSplit() {
      const res = await fetch('/api/export_split', { method: 'POST' });
      const data = await res.json();
      showToast('Clean Split exported to ' + data.path);
    }

    async function exportMaterializedDataset() {
      showToast('Starting clean dataset export (this runs in background)...');
      const res = await fetch('/api/export_dataset', { method: 'POST' });
      const data = await res.json();
      showToast('Exported dataset to ' + data.path);
    }

    function showToast(msg) {
      const t = document.getElementById('toast-msg');
      t.innerText = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 3000);
    }

    window.onload = init;
  </script>
</body>
</html>
"""


class CurationHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the reactive Curation UI and JSON REST endpoints."""

    curator: DatasetCurator
    manifest: CurationManifest
    manifest_path: Path
    output_dir: Path

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/manifest":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.manifest.to_dict()).encode("utf-8"))
            return

        if path.startswith("/api/episode/"):
            try:
                ep_idx = int(path.split("/")[-1])
                trajectories = self.curator.get_episode_trajectories(ep_idx)
                payload = {
                    "episode_index": ep_idx,
                    "states": trajectories["states"].tolist(),
                    "actions": trajectories["actions"].tolist(),
                    "timestamps": trajectories["timestamps"].tolist(),
                }
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                return
            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
                return

        if path.startswith("/api/video/"):
            try:
                ep_idx = int(path.split("/")[-1])
                self._stream_video(ep_idx)
                return
            except Exception as e:
                self.send_error(HTTPStatus.NOT_FOUND, str(e))
                return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        if path == "/api/update_episode":
            data = json.loads(body)
            ep_idx = data["episode_index"]
            self.manifest.mark_episode(
                episode_index=ep_idx,
                status=data.get("status"),
                trim_start=data.get("trim_start"),
                trim_end=data.get("trim_end"),
                tags=data.get("tags"),
                notes=data.get("notes"),
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            return

        if path == "/api/save_manifest":
            self.manifest.save_json(self.manifest_path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "path": str(self.manifest_path)}).encode("utf-8"))
            return

        if path == "/api/export_split":
            out_file = self.output_dir / "clean_split.json"
            self.curator.export_split_manifest(self.manifest, out_file)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "path": str(out_file)}).encode("utf-8"))
            return

        if path == "/api/export_dataset":
            clean_root = self.output_dir / "curated_clean_dataset"
            clean_repo_id = f"{self.curator.repo_id}_curated"
            self.curator.export_clean_dataset(
                manifest=self.manifest,
                output_repo_id=clean_repo_id,
                output_root=clean_root,
                slice_frames=True,
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "path": str(clean_root)}).encode("utf-8"))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _stream_video(self, ep_idx: int) -> None:
        """Streams the video file with HTTP 206 Partial Content (byte range requests)."""
        meta = self.curator.dataset.meta
        cam = self.curator.camera_key
        if not cam:
            self.send_error(HTTPStatus.NOT_FOUND, "No camera found in dataset")
            return

        chunk = meta.episodes[f"videos/{cam}/chunk_index"][ep_idx]
        file_idx = meta.episodes[f"videos/{cam}/file_index"][ep_idx]
        vpath = self.curator.dataset.root / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"

        if not vpath.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Video file not found at {vpath}")
            return

        file_size = vpath.stat().st_size
        range_header = self.headers.get("Range", None)

        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(vpath, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(vpath, "rb") as f:
            self.wfile.write(f.read())


def start_curation_server(
    curator: DatasetCurator,
    manifest: CurationManifest,
    manifest_path: Path | str,
    output_dir: Path | str,
    port: int = 8080,
    open_browser: bool = False,
    host: str = "127.0.0.1",
) -> None:
    """Launches the interactive curation HTTP server."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = type(
        "ConfiguredCurationHandler",
        (CurationHTTPHandler,),
        {
            "curator": curator,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "output_dir": output_dir,
        },
    )

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://localhost:{port}"
        logger.info(f"Dataset Curation UI running at: {url}")
        print("\n=======================================================")
        print(f"🚀 LeRobot Dataset Curation UI active at: {url}")
        print("Press Ctrl+C to stop the server.")
        print("=======================================================\n")

        if open_browser:
            import webbrowser

            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down curation server...")
            httpd.shutdown()
