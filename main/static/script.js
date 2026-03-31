document.addEventListener("DOMContentLoaded", () => {

  // =====================
  // NAV
  // =====================
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => navbar.classList.toggle('scrolled', window.scrollY > 20));

  document.getElementById('hamburger').addEventListener('click', () => {
    document.getElementById('navLinks').classList.toggle('open');
  });

  document.querySelectorAll('.nav-links a').forEach(a => {
    a.addEventListener('click', () => document.getElementById('navLinks').classList.remove('open'));
  });

  // =====================
  // UPLOAD
  // =====================
  let selectedFile = null;
  const dropZone   = document.getElementById('dropZone');
  const fileInput  = document.getElementById('fileInput');
  const fileInfo   = document.getElementById('fileInfo');
  const predictBtn = document.getElementById('predictBtn');

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('active');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('active');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  document.getElementById('removeFile').addEventListener('click', () => {
    selectedFile = null;
    fileInfo.style.display = 'none';
    predictBtn.disabled = true;
    fileInput.value = '';
  });

  function handleFile(f) {
    if (!f.name.endsWith('.csv')) {
      showToast('Only CSV files are allowed', 'error');
      return;
    }
    selectedFile = f;
    document.getElementById('fileName').textContent = f.name;
    document.getElementById('fileSize').textContent = (f.size / 1024).toFixed(1) + ' KB';
    fileInfo.style.display = 'flex';
    predictBtn.disabled = false;
    showToast('File "' + f.name + '" ready for analysis', 'success');
  }

  // =====================
  // PREDICT
  // =====================
  let resultsData = [];

  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<div class="spinner"></div> Analyzing...';

    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      const res = await fetch('http://127.0.0.1:8000/predict-batch', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('fail');
      const data = await res.json();
      const rows = Array.isArray(data) ? data : data.predictions || data.results || [];
      processResults(rows);
      showToast('Prediction complete!', 'success');
    } catch {
      // Demo fallback when backend is unavailable
      const demo = Array.from({ length: 20 }, (_, i) => ({
        id: i + 1,
        amount: +(Math.random() * 5000).toFixed(2),
        prediction: Math.random() > .8 ? 1 : 0
      }));
      processResults(demo);
      showToast('Using demo data (backend unavailable)', 'info');
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = '✓ Predict Fraud';
    }
  });

  function processResults(rows) {
    resultsData = rows;
    const fraud = rows.filter(r => r.prediction === 1 || r.is_fraud === true || r.is_fraud === 1).length;
    const total = rows.length;
    const pct   = total ? (fraud / total * 100).toFixed(2) : 0;

    document.getElementById('totalCount').textContent = total;
    document.getElementById('fraudCount').textContent = fraud;
    document.getElementById('fraudRate').textContent  = pct + '%';

    // Build table
    const keys = Object.keys(rows[0] || {});
    let html = '<table><thead><tr>' + keys.map(k => '<th>' + k + '</th>').join('') + '</tr></thead><tbody>';

    rows.forEach(r => {
      const isFraud = r.prediction === 1 || r.is_fraud === true || r.is_fraud === 1;
      html += '<tr class="' + (isFraud ? 'fraud-row' : 'safe-row') + '">';
      keys.forEach(k => {
        let v = r[k];
        if ((k === 'prediction' || k === 'is_fraud') && typeof v === 'number') v = v === 1 ? 'Fraud' : 'Legit';
        html += '<td>' + v + '</td>';
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    document.getElementById('tableWrap').innerHTML = html;

    document.getElementById('results').classList.add('visible');
    document.getElementById('charts').classList.add('visible');
    renderCharts(total - fraud, fraud);
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
  }

  // =====================
  // DOWNLOAD
  // =====================
  document.getElementById('downloadBtn').addEventListener('click', () => {
    if (!resultsData.length) return;
    const keys = Object.keys(resultsData[0]);
    let csv = keys.join(',') + '\n';
    resultsData.forEach(r => { csv += keys.map(k => r[k]).join(',') + '\n'; });
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'fraud_results.csv';
    a.click();
  });

  // =====================
  // CHARTS
  // =====================
  let dChart, bChart;

  function renderCharts(legit, fraud) {
    if (dChart) dChart.destroy();
    if (bChart) bChart.destroy();

    const colors = ['#38bdf8', '#ef4444'];

    dChart = new Chart(document.getElementById('doughnutChart'), {
      type: 'doughnut',
      data: {
        labels: ['Legitimate', 'Fraudulent'],
        datasets: [{
          data: [legit, fraud],
          backgroundColor: colors,
          borderColor: ['#0ea5e9', '#dc2626'],
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#94a3b8' } } }
      }
    });

    bChart = new Chart(document.getElementById('barChart'), {
      type: 'bar',
      data: {
        labels: ['Legitimate', 'Fraudulent'],
        datasets: [{
          label: 'Transactions',
          data: [legit, fraud],
          backgroundColor: ['rgba(56,189,248,0.7)', 'rgba(239,68,68,0.7)'],
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#94a3b8' } } },
        scales: {
          x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(30,41,59,0.5)' } },
          y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(30,41,59,0.5)' } }
        }
      }
    });
  }

  // =====================
  // TOAST
  // =====================
  function showToast(msg, type) {
    const t = document.createElement('div');
    t.className = 'toast ' + type;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3000);
  }

});