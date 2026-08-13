/**
 * DataPilot AI – AI CSV Analyst
 * Frontend Application Logic (Flask + Pandas Backend Integration)
 */

const API_BASE_URL = "";

// Global App State
const appState = {
    isUploaded: false,
    fileName: '',
    data: null
};

const chartInstances = {};
let activeChartTarget = 'query-result-chart';

// Application Initialization
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupEventListeners();
});

function setupEventListeners() {
    // Theme Toggle
    document.getElementById('theme-toggle')?.addEventListener('click', toggleDarkMode);

    // CSV File Selection Preview
    const fileInput = document.getElementById('csv-file-input');
    fileInput?.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            document.getElementById('upload-status').innerHTML = file.name.endsWith('.csv')
                ? `<p style="color: var(--accent-color);">Selected: <strong>${escapeHTML(file.name)}</strong></p>`
                : `<p style="color: #ef4444;">Please select a valid .csv file.</p>`;
        }
    });

    // Upload Form Submission
    document.getElementById('upload-form')?.addEventListener('submit', handleFileUpload);

    // Query Form Submission ("Ask Your Data")
    document.getElementById('query-form')?.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = document.getElementById('user-query')?.value.trim() || '';
        handleQuerySubmit(query);
    });

    // Example Query Chips
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const queryInput = document.getElementById('user-query');
            if (queryInput) queryInput.value = chip.textContent.replace(/^"|"$/g, '');
        });
    });
}

// ==========================================
// 1. DARK MODE MANAGEMENT
// ==========================================
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('datapilot_theme', isDark ? 'dark' : 'light');
    updateThemeBtn(isDark);
}

function initTheme() {
    const savedTheme = localStorage.getItem('datapilot_theme');
    const isDark = savedTheme === 'dark';
    if (isDark) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
    updateThemeBtn(isDark);
}

function updateThemeBtn(isDark) {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.querySelector('.theme-icon').textContent = isDark ? '☀️' : '🌙';
    btn.querySelector('.theme-text').textContent = isDark ? 'Light Mode' : 'Dark Mode';
}

// ==========================================
// 2. CSV UPLOAD API CALL (POST /api/upload)
// ==========================================
async function handleFileUpload(e) {
    e.preventDefault();
    const fileInput = document.getElementById('csv-file-input');
    const file = fileInput?.files[0];

    if (!file) return showError('Please select a CSV file first.', 'upload-status');
    if (!file.name.endsWith('.csv')) return showError('Only .csv files are supported.', 'upload-status');

    setAnalyzing(true, 'csv-upload');
    showStatus('Uploading and analyzing dataset on Flask backend...', 'upload-status');

    const formData = new FormData();
    formData.append('csv-file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Server error processing CSV file.');
        }

        // Store active data from Flask API
        appState.isUploaded = true;
        appState.fileName = data.filename;
        appState.data = data;

        // Render UI sections using Flask JSON data
        updateOverview(data);
        updatePreviewTable(data);
        performEDA(data);
        generateInsights(data);

        showStatus(`✓ Dataset "${escapeHTML(data.filename)}" successfully analyzed by Pandas backend!`, 'upload-status', '#10b981');

    } catch (err) {
        showError('Backend API Error: ' + err.message, 'upload-status');
    } finally {
        setAnalyzing(false, 'csv-upload');
    }
}

// ==========================================
// 3. UI DASHBOARD RENDERING
// ==========================================
function updateOverview(data) {
    setText('metric-rows', data.total_rows.toLocaleString());
    setText('metric-columns', data.total_cols.toLocaleString());
    setText('metric-missing', data.total_missing.toLocaleString());
    setText('metric-numeric', data.numeric_columns.length.toLocaleString());
    setText('metric-categorical', data.categorical_columns.length.toLocaleString());
}

function updatePreviewTable(data) {
    const head = document.getElementById('table-header-row');
    const body = document.getElementById('table-body');
    if (!head || !body) return;

    // Table Headers
    head.innerHTML = data.columns.map(col => `<th>${escapeHTML(col)}</th>`).join('');

    // Table Rows (from Flask preview_data)
    body.innerHTML = data.preview_data.map(row => {
        const cells = data.columns.map(col => `<td>${escapeHTML(row[col] !== undefined ? row[col] : '')}</td>`).join('');
        return `<tr>${cells}</tr>`;
    }).join('');
}

function performEDA(data) {
    const container = document.getElementById('summary-stats-container');
    if (!container) return;

    let html = `<div class="table-wrapper"><table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
        <thead><tr style="background: var(--bg-color); text-align: left;">
            <th style="padding: 0.6rem;">Column</th><th style="padding: 0.6rem;">Type</th>
            <th style="padding: 0.6rem;">Mean</th><th style="padding: 0.6rem;">Min</th>
            <th style="padding: 0.6rem;">Max</th><th style="padding: 0.6rem;">Missing</th>
            <th style="padding: 0.6rem;">Unique</th>
        </tr></thead><tbody>`;

    data.columns.forEach(col => {
        const stats = data.eda_summary[col] || {};
        html += `<tr style="border-bottom: 1px solid var(--border-color);">
            <td style="padding: 0.6rem; font-weight: 600;">${escapeHTML(col)}</td>
            <td style="padding: 0.6rem;">${stats.is_numeric ? 'Numeric' : 'Categorical'}</td>
            <td style="padding: 0.6rem;">${stats.mean}</td>
            <td style="padding: 0.6rem;">${stats.min}</td>
            <td style="padding: 0.6rem;">${stats.max}</td>
            <td style="padding: 0.6rem;">${stats.missing}</td>
            <td style="padding: 0.6rem;">${stats.unique}</td>
        </tr>`;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;

    renderMissingValuesChart(data);
    renderDistributionChart(data);
    renderCorrelationChart(data);
}

function createChart(type, labels, values, title) {
    const container = document.getElementById(activeChartTarget);
    if (!container) return;

    if (!window.Chart) {
        container.innerHTML = '<p>Chart library unavailable.</p>';
        return;
    }

    if (!labels || !labels.length || !values || !values.length || values.some(value => Number.isNaN(Number(value)))) {
        container.innerHTML = '<p>No chart data available for this result.</p>';
        return;
    }

    if (chartInstances[activeChartTarget]) chartInstances[activeChartTarget].destroy();

    // Preserve any note or supplementary content inside the container
    // Remove any existing canvas and re-create it so surrounding info is not lost.
    const existingCanvas = container.querySelector('canvas');
    if (existingCanvas) existingCanvas.remove();
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    if (!canvas) return;

    const isDark = document.body.classList.contains('dark-mode');
    const textColor = isDark ? '#e2e8f0' : '#0f172a';
    const gridColor = isDark ? 'rgba(148, 163, 184, 0.2)' : 'rgba(100, 116, 139, 0.2)';
    const colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#f97316'];

    const dataset = {
        label: title,
        data: values,
        borderColor: colors[0],
        backgroundColor: colors.map((c, i) => i === 0 ? c + 'cc' : c + 'aa'),
        borderWidth: 2,
        fill: type === 'line' ? false : true,
        tension: 0.3,
        pointRadius: type === 'line' ? 4 : 0
    };

    // Short-format number like 12.5K, 1.2M while keeping tooltips showing full value
    function formatShortNumber(n) {
        if (n === null || n === undefined || Number.isNaN(Number(n))) return '';
        const num = Number(n);
        const abs = Math.abs(num);
        if (abs >= 1e9) return (num / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
        if (abs >= 1e6) return (num / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
        if (abs >= 1e3) return (num / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
        return num.toString();
    }

    // Plugin to draw value labels above each bar so ensures visibility for small bars
    const valueLabelPlugin = {
        id: 'valueLabelPlugin',
        afterDatasetsDraw: (chart) => {
            const ctx = chart.ctx;
            const chartArea = chart.chartArea;
            ctx.save();
            ctx.textBaseline = 'middle';

            chart.data.datasets.forEach((ds, datasetIndex) => {
                const meta = chart.getDatasetMeta(datasetIndex);
                if (!meta || !meta.data) return;

                meta.data.forEach((elem, index) => {
                    try {
                        const value = ds.data[index];
                        if (value === null || value === undefined || Number.isNaN(Number(value))) return;
                        const display = formatShortNumber(value);

                        const x = elem.x;
                        // For vertical bars element.y is the top of the bar for positive values
                        const y = elem.y;

                        // Text style
                        const isDark = document.body.classList.contains('dark-mode');
                        const textColor = isDark ? '#0f172a' : '#0f172a'; // dark text on both for readability
                        ctx.fillStyle = textColor;
                        ctx.font = '600 12px Inter, Roboto, Arial, sans-serif';
                        ctx.textAlign = 'center';

                        const metrics = ctx.measureText(display);
                        const textHeight = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent || 12;
                        const offset = 8;

                        // Proposed label position above the bar
                        let labelY = y - offset - textHeight / 2;

                        // If label would be above chart top, place it below the bar
                        if (labelY < chartArea.top + 2) {
                            const bottom = (typeof elem.base !== 'undefined') ? elem.base : (elem.y + (elem.height || 0));
                            labelY = bottom + offset + textHeight / 2;
                        }

                        // Draw a light halo to keep text readable on colored bars
                        ctx.lineWidth = 3;
                        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
                        ctx.strokeText(display, x, labelY);
                        ctx.fillText(display, x, labelY);
                    } catch (e) {
                        // ignore per-bar labeling errors
                    }
                });
            });

            ctx.restore();
        }
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: textColor }
            },
            title: {
                display: true,
                text: title,
                color: textColor,
                font: { size: 14, weight: '600' }
            }
        },
        scales: type === 'pie' ? {} : {
            x: {
                ticks: { color: textColor },
                grid: { color: gridColor, drawBorder: false },
                display: type !== 'pie'
            },
            y: {
                ticks: { color: textColor },
                grid: { color: gridColor, drawBorder: false }
            }
        }
    };

    chartInstances[activeChartTarget] = new Chart(canvas, {
        type,
        data: {
            labels,
            datasets: [dataset]
        },
        options,
        plugins: [valueLabelPlugin]
    });
}

function renderMissingValuesChart(data) {
    const missingChart = document.getElementById('missing-values-chart');
    if (!missingChart) return;

    const missingCols = Object.entries(data.missing_values || {})
        .map(([name, missing]) => ({ name, missing }))
        .filter(item => item.missing > 0)
        .sort((a, b) => b.missing - a.missing)
        .slice(0, 5);

    if (missingCols.length === 0) {
        missingChart.innerHTML = `
            <div style="text-align: center; padding: 1.25rem; color: #10b981;">
                <span style="font-size: 1.5rem;">✓</span>
                <p style="font-weight: 600; margin-top: 0.25rem;">No Missing Values</p>
                <p style="font-size: 0.8rem; color: var(--text-muted);">All ${data.total_rows.toLocaleString()} rows in this dataset are 100% complete.</p>
            </div>`;
        return;
    }

    // Clear any previous placeholder text/notes before rendering the chart
    missingChart.innerHTML = '';
    activeChartTarget = 'missing-values-chart';
    createChart('bar', missingCols.map(item => item.name), missingCols.map(item => item.missing), 'Missing Values');
}

function renderDistributionChart(data) {
    const distributionChart = document.getElementById('distribution-chart');
    if (!distributionChart) return;

    const distributions = data.distribution_data || {};
    const distributionEntry = Object.entries(distributions)
        .find(([, distribution]) => distribution.labels?.length > 1 && distribution.values?.length > 1)
        || Object.entries(distributions)
            .find(([, distribution]) => distribution.labels?.length && distribution.values?.length);

    if (!distributionEntry) {
        distributionChart.innerHTML = '<p>No numeric columns available for distribution chart.</p>';
        return;
    }

    const [column, distribution] = distributionEntry;
    // Clear any previous placeholder text/notes before rendering the chart
    distributionChart.innerHTML = '';
    activeChartTarget = 'distribution-chart';
    createChart('bar', distribution.labels, distribution.values, `Distribution of ${column}`);
}

function renderCorrelationChart(data) {
    const correlationChart = document.getElementById('correlation-chart');
    if (!correlationChart) return;

    const correlation = data.correlation_data || {};
    const columns = correlation.columns || [];
    const matrix = correlation.matrix || [];
    const excluded = data.excluded_identifier_columns || [];
    if (columns.length < 2 || matrix.length < 2) {
        if (excluded && excluded.length) {
            correlationChart.innerHTML = `<p>Excluded identifier columns from correlation analysis: <strong>${escapeHTML(excluded.join(', '))}</strong></p>`;
            correlationChart.insertAdjacentHTML('beforeend', '<p>Not enough numeric columns remain to compute correlations.</p>');
        } else {
            correlationChart.innerHTML = '<p>Correlation chart needs at least 2 numeric columns.</p>';
        }
        return;
    }

    const pairs = [];
    for (let row = 0; row < columns.length; row += 1) {
        for (let col = row + 1; col < columns.length; col += 1) {
            const value = matrix[row]?.[col];
            if (typeof value === 'number' && Number.isFinite(value)) {
                // Only use the upper triangle: no self-correlations or duplicates.
                pairs.push({ left: columns[row], right: columns[col], value });
            }
        }
    }

    if (!pairs.length) {
        correlationChart.innerHTML = '<p>Correlation could not be calculated because the numeric columns have no varying values.</p>';
        return;
    }

    if (chartInstances['correlation-chart']) {
        chartInstances['correlation-chart'].destroy();
        delete chartInstances['correlation-chart'];
    }

    const count = columns.length;
    const mode = count <= 10 ? 'compact' : count <= 20 ? 'expanded' : 'large';
    const cellSize = count <= 10 ? 58 : count <= 20 ? 50 : 34;
    const annotationSize = count <= 10 ? 12 : count <= 20 ? 10 : 0;

    correlationChart.innerHTML = '';
    correlationChart.classList.add('correlation-visualization');
    if (excluded && excluded.length) {
        const note = document.createElement('p');
        note.className = 'correlation-note';
        note.textContent = `Excluded identifier columns: ${excluded.join(', ')}`;
        correlationChart.appendChild(note);
    }

    const intro = document.createElement('div');
    intro.className = 'correlation-intro';
    intro.innerHTML = `<strong>Correlation heatmap</strong><span>${count} numeric variables${mode === 'large' ? ' · scroll to explore the full matrix' : ''}</span>`;
    correlationChart.appendChild(intro);

    const scrollArea = document.createElement('div');
    scrollArea.className = `correlation-heatmap-scroll correlation-heatmap--${mode}`;
    scrollArea.setAttribute('tabindex', '0');
    scrollArea.setAttribute('aria-label', `Correlation heatmap for ${count} numeric columns`);

    const grid = document.createElement('div');
    grid.className = 'correlation-heatmap-grid';
    grid.style.setProperty('--column-count', count);
    grid.style.setProperty('--cell-size', `${cellSize}px`);
    grid.style.setProperty('--annotation-size', `${annotationSize}px`);

    const corner = document.createElement('div');
    corner.className = 'correlation-heatmap-corner';
    grid.appendChild(corner);
    columns.forEach(column => {
        const label = document.createElement('div');
        label.className = 'correlation-column-label';
        label.textContent = column;
        label.title = column;
        grid.appendChild(label);
    });

    columns.forEach((rowName, rowIndex) => {
        const rowLabel = document.createElement('div');
        rowLabel.className = 'correlation-row-label';
        rowLabel.textContent = rowName;
        rowLabel.title = rowName;
        grid.appendChild(rowLabel);

        columns.forEach((columnName, columnIndex) => {
            const value = matrix[rowIndex]?.[columnIndex];
            const cell = document.createElement('div');
            cell.className = 'correlation-cell';
            if (typeof value === 'number' && Number.isFinite(value)) {
                cell.style.backgroundColor = correlationColor(value);
                if (Math.abs(value) < 0.45) cell.style.color = '#0f172a';
                cell.title = `${rowName} ↔ ${columnName}: ${value.toFixed(2)}`;
                cell.setAttribute('aria-label', cell.title);
                if (annotationSize) cell.textContent = value.toFixed(2);
            } else {
                cell.classList.add('correlation-cell--empty');
                cell.title = `${rowName} ↔ ${columnName}: unavailable`;
            }
            grid.appendChild(cell);
        });
    });
    scrollArea.appendChild(grid);
    correlationChart.appendChild(scrollArea);

    const legend = document.createElement('div');
    legend.className = 'correlation-legend';
    legend.innerHTML = '<span>-1</span><div class="correlation-legend-gradient" aria-hidden="true"></div><span>0</span><div class="correlation-legend-gradient correlation-legend-gradient--positive" aria-hidden="true"></div><span>+1</span>';
    correlationChart.appendChild(legend);
    correlationChart.appendChild(createStrongCorrelationSummary(pairs));
}

function correlationColor(value) {
    const clamped = Math.max(-1, Math.min(1, value));
    const magnitude = Math.abs(clamped);
    const base = clamped >= 0 ? [37, 99, 235] : [225, 29, 72];
    const alpha = 0.12 + magnitude * 0.82;
    return `rgba(${base[0]}, ${base[1]}, ${base[2]}, ${alpha})`;
}

function createStrongCorrelationSummary(pairs) {
    const summary = document.createElement('section');
    summary.className = 'strong-correlation-summary';

    const heading = document.createElement('h4');
    heading.textContent = 'Strongest Relationships';
    summary.appendChild(heading);

    const description = document.createElement('p');
    description.textContent = 'Unique pairs, ranked by correlation strength.';
    summary.appendChild(description);

    const lists = document.createElement('div');
    lists.className = 'strong-correlation-lists';
    const positive = pairs.filter(pair => pair.value > 0).sort((a, b) => b.value - a.value).slice(0, 5);
    const negative = pairs.filter(pair => pair.value < 0).sort((a, b) => a.value - b.value).slice(0, 5);

    lists.appendChild(createCorrelationList('Positive', positive, 'positive'));
    lists.appendChild(createCorrelationList('Negative', negative, 'negative'));
    summary.appendChild(lists);
    return summary;
}

function createCorrelationList(title, pairs, tone) {
    const group = document.createElement('div');
    group.className = `strong-correlation-group strong-correlation-group--${tone}`;
    const heading = document.createElement('h5');
    heading.textContent = title;
    group.appendChild(heading);
    const list = document.createElement('ul');
    if (!pairs.length) {
        const item = document.createElement('li');
        item.textContent = 'No relationships found.';
        list.appendChild(item);
    } else {
        pairs.forEach(pair => {
            const item = document.createElement('li');
            const names = document.createElement('span');
            names.textContent = `${pair.left} ↔ ${pair.right}`;
            const value = document.createElement('strong');
            value.textContent = pair.value.toFixed(2);
            item.append(names, value);
            list.appendChild(item);
        });
    }
    group.appendChild(list);
    return group;
}

function generateInsights(data) {
    const container = document.getElementById('ai-insights-container');
    if (container) {
        container.innerHTML = `
            <div style="line-height: 1.6;">
                <p style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-main);">Dataset Summary from Flask Backend:</p>
                <ul style="padding-left: 1.25rem; font-size: 0.9rem; color: var(--text-muted);">
                    <li>Loaded dataset <strong>"${escapeHTML(data.filename)}"</strong> with <strong>${data.total_rows.toLocaleString()}</strong> rows and <strong>${data.total_cols}</strong> columns.</li>
                    <li>Detected <strong>${data.numeric_columns.length}</strong> numerical metrics and <strong>${data.categorical_columns.length}</strong> categorical features.</li>
                    <li>Pandas backend engine is ready to run live AI queries and analytical aggregations.</li>
                </ul>
            </div>`;
    }
}

// ==========================================
// 4. ASK YOUR DATA API CALL 
// ==========================================
async function handleQuerySubmit(query) {
    if (!appState.isUploaded) return showError('Please upload a CSV dataset before asking questions.', 'query-result-text');
    if (!query) return showError('Please enter a question about your dataset.', 'query-result-text');

    setAnalyzing(true, 'ask-data');
    showStatus(`⏳ Analyzing query: "<em>${escapeHTML(query)}</em>"...`, 'query-result-text', 'var(--accent-color)');

    try {
        const response = await fetch(`${API_BASE_URL}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Server error processing your question.');
        }

        const resultBox = document.getElementById('query-result-text');
        if (resultBox) {
            resultBox.innerHTML = `
                <div style="line-height: 1.6;">
                    <p style="font-weight: 600; color: var(--accent-color); margin-bottom: 0.4rem;">Pandas & AI Answer:</p>
                    <p style="font-size: 0.95rem; color: var(--text-main);">${escapeHTML(data.answer)}</p>
                </div>`;
        }

        // Prefer the explicit chart plan returned by the backend. The legacy
        // data-based inference remains as a fallback for older API responses.
        const chartData = Array.isArray(data.charts) && data.charts.length
            ? data.charts[0]
            : extractChartData(data);
        if (chartData) {
            activeChartTarget = 'query-result-chart';
            createChart(chartData.type, chartData.labels, chartData.values, chartData.title);
        }
    } catch (err) {
        showError('Query Analysis Error: ' + err.message, 'query-result-text');
    } finally {
        setAnalyzing(false, 'ask-data');
    }
}

function extractChartData(response) {
    const rows = Array.isArray(response?.data) ? response.data : [];
    if (!rows.length) return null;

    const keyNames = Object.keys(rows[0]);
    const labelKey = keyNames.find(key => !['value', 'total', 'sum', 'average', 'count', 'minimum', 'maximum', 'revenue', 'amount'].includes(key.toLowerCase())) || keyNames[0];
    const valueKey = keyNames.find(key => ['value', 'total', 'sum', 'average', 'count', 'minimum', 'maximum', 'revenue', 'amount'].includes(key.toLowerCase())) || keyNames[1] || keyNames[0];

    const labels = rows.map(row => row[labelKey] ?? 'Item');
    const values = rows.map(row => Number(row[valueKey] ?? 0));

    if (!values.length || values.some(value => Number.isNaN(value))) return null;

    const type = pickChartTypeFromQuery(document.getElementById('user-query')?.value || '');
    return {
        type,
        labels,
        values,
        title: valueKey.charAt(0).toUpperCase() + valueKey.slice(1)
    };
}

function pickChartTypeFromQuery(queryText) {
    const query = (queryText || '').toLowerCase();
    if (query.includes('trend') || query.includes('month') || query.includes('year') || query.includes('time') || query.includes('over time') || query.includes('day')) return 'line';
    if (query.includes('share') || query.includes('proportion') || query.includes('percentage') || query.includes('distribution') || query.includes('breakdown')) return 'pie';
    return 'bar';
}

// ==========================================
// 5. HELPER UTILITIES
// ==========================================
function setAnalyzing(isAnalyzing, elementId) {
    const elem = document.getElementById(elementId);
    if (!elem) return;
    elem.classList.toggle('analyzing', isAnalyzing);
    elem.querySelectorAll('button').forEach(btn => btn.disabled = isAnalyzing);
}

function showError(message, containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<p style="color: #ef4444; font-weight: 500;">⚠️ ${escapeHTML(message)}</p>`;
    }
}

function showStatus(message, containerId, color) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<p style="color: ${color || 'var(--accent-color)'}; font-weight: 500;">${message}</p>`;
    }
}

function setText(cardId, val) {
    const elem = document.getElementById(cardId)?.querySelector('.metric-value');
    if (elem) elem.textContent = val;
}

function escapeHTML(str) {
    return str !== undefined && str !== null ? String(str).replace(/[&<>"']/g, m => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;' }[m])) : '';
}
