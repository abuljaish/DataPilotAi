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

let activeChart = null;
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

    if (activeChart) activeChart.destroy();
    container.innerHTML = '<canvas></canvas>';

    const canvas = container.querySelector('canvas');
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

    activeChart = new Chart(canvas, {
        type,
        data: {
            labels,
            datasets: [dataset]
        },
        options
    });
}

function renderMissingValuesChart(data) {
    const missingChart = document.getElementById('missing-values-chart');
    if (!missingChart) return;

    const missingCols = data.columns
        .map(col => ({ name: col, missing: data.eda_summary[col]?.missing || 0 }))
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

    activeChartTarget = 'missing-values-chart';
    createChart('bar', missingCols.map(item => item.name), missingCols.map(item => item.missing), 'Missing Values');
}

function renderDistributionChart(data) {
    const distributionChart = document.getElementById('distribution-chart');
    if (!distributionChart) return;

    const numericCols = (data.numeric_columns || []).slice(0, 6);
    if (!numericCols.length) {
        distributionChart.innerHTML = '<p>No numeric columns available for distribution chart.</p>';
        return;
    }

    if (numericCols.length === 1) {
        distributionChart.innerHTML = `
            <div style="padding: 1rem; width: 100%; text-align: left;">
                <p style="font-weight: 600; margin-bottom: 0.5rem;">Numeric Distribution Summary</p>
                <p style="margin: 0;">${escapeHTML(numericCols[0])}: mean = ${data.eda_summary[numericCols[0]]?.mean ?? 'N/A'}</p>
            </div>`;
        return;
    }

    activeChartTarget = 'distribution-chart';
    const labels = numericCols;
    const values = numericCols.map(col => Number(data.eda_summary[col]?.mean || 0));
    createChart('bar', labels, values, 'Average Value by Numeric Column');
}

function renderCorrelationChart(data) {
    const correlationChart = document.getElementById('correlation-chart');
    if (!correlationChart) return;

    const numericCols = (data.numeric_columns || []).slice(0, 6);
    if (numericCols.length < 2) {
        correlationChart.innerHTML = '<p>Correlation chart needs at least 2 numeric columns.</p>';
        return;
    }

    const values = numericCols.map(col => Number(data.eda_summary[col]?.mean || 0));
    const labels = numericCols;

    activeChartTarget = 'correlation-chart';
    createChart('bar', labels, values, 'Numeric Column Means');
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
// 4. ASK YOUR DATA API CALL (POST /api/ask)
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

        const chartData = extractChartData(data);
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
