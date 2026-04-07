let myChart = null;
        let statusInterval = null;

        const loadingTexts = [
            "Menganalisa Struktur Sel...",
            "Membedah Variabel Biologis...",
            "Mengukur Skor Malignansi...",
            "Menghitung Probabilitas Karsinogenik...",
            "Melakukan Validasi Neural...",
            "Menyiapkan Ringkasan Klinis..."
        ];

        function setLoader(show, title = "AI ENGINE ACTIVE") {
            const overlay = document.getElementById('loadingOverlay');
            const statusEl = document.getElementById('loadingStatus');
            const titleEl = document.getElementById('loadingTitle');
            
            if(show) {
                titleEl.innerText = title;
                overlay.classList.remove('pointer-events-none', 'opacity-0');
                overlay.classList.add('opacity-100');
                
                let i = 0;
                statusInterval = setInterval(() => {
                    statusEl.innerText = loadingTexts[i % loadingTexts.length];
                    i++;
                }, 400);
            } else {
                overlay.classList.add('pointer-events-none', 'opacity-0');
                overlay.classList.remove('opacity-100');
                clearInterval(statusInterval);
            }
        }

        function switchTab(mode) {
            const isJson = mode === 'json';
            document.getElementById('section-json').classList.toggle('hidden', !isJson);
            document.getElementById('section-csv').classList.toggle('hidden', isJson);
            
            document.getElementById('tab-json').className = isJson ? 
                "px-8 py-2.5 rounded-xl text-[11px] font-black bg-sky-600 text-white shadow-lg shadow-sky-900/40 tracking-widest" : 
                "px-8 py-2.5 rounded-xl text-[11px] font-black text-slate-400 tracking-widest";
            
            document.getElementById('tab-csv').className = !isJson ? 
                "px-8 py-2.5 rounded-xl text-[11px] font-black bg-sky-600 text-white shadow-lg shadow-sky-900/40 tracking-widest" : 
                "px-8 py-2.5 rounded-xl text-[11px] font-black text-slate-400 tracking-widest";
        }

        document.getElementById('csvFile').addEventListener('change', e => {
            if(e.target.files[0]) {
                document.getElementById('fileName').innerText = e.target.files[0].name.toUpperCase();
                document.getElementById('fileName').className = "text-xs font-bold text-sky-400 uppercase tracking-tighter";
            }
        });

        async function sendJsonPrediction() {
            const input = document.getElementById('jsonInput').value;
            if(!input) return;

            setLoader(true, "NEURAL PROCESSING");
            try {
                const res = await fetch('http://127.0.0.1:8000/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: input
                });
                const data = await res.json();
                
                setTimeout(() => {
                    showResult('json');
                    const isMal = data.result === 'Malignant';
                    const p = isMal ? data.confidence_score.malignant : data.confidence_score.benign;
                    
                    document.getElementById('labelResult').innerText = isMal ? "GANAS" : "JINAK";
                    document.getElementById('labelResult').className = `text-8xl font-black italic tracking-tighter mb-8 leading-none ${isMal ? 'neon-text-red' : 'neon-text-green'}`;
                    document.getElementById('probValue').innerText = p + '%';
                    document.getElementById('probBar').style.width = p + '%';
                    document.getElementById('probBar').className = `h-full rounded-full transition-all duration-1000 ${isMal ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'}`;
                    
                    document.getElementById('statTotal').innerText = "1";
                    document.getElementById('statMalignant').innerText = isMal ? "1" : "0";
                    document.getElementById('statBenign').innerText = isMal ? "0" : "1";
                    document.getElementById('statConfidence').innerText = p + "%";
                    document.getElementById('tableBadge').innerText = "SINGLE_LOG";
                    
                    document.getElementById('csvTableBody').innerHTML = `
                        <tr class="bg-slate-900/20">
                            <td class="p-4 text-slate-400">PASIEN-001</td>
                            <td class="p-4 font-bold ${isMal ? 'text-red-400' : 'text-emerald-400'}">${isMal ? 'GANAS' : 'JINAK'}</td>
                            <td class="p-4 text-right text-slate-300">${p}%</td>
                        </tr>
                    `;
                    setLoader(false);
                }, 2000); // Sedikit lebih lama untuk memamerkan animasi
            } catch (e) { 
                setLoader(false);
                alert("Koneksi API Gagal!"); 
            }
        }

        async function sendCsvPrediction() {
            const fileInput = document.getElementById('csvFile');
            if (!fileInput.files[0]) return;

            setLoader(true, "BATCH PROTOCOL ACTIVE");
            const fd = new FormData();
            fd.append('file', fileInput.files[0]);
            
            try {
                const res = await fetch('http://127.0.0.1:8000/predict-csv', { method: 'POST', body: fd });
                const data = await res.json();
                
                setTimeout(() => {
                    showResult('csv');
                    let mCount = data.predictions.filter(x => x.prediction === 'Malignant').length;
                    let bCount = data.total_rows - mCount;
                    let avgConf = (data.predictions.reduce((acc, curr) => acc + parseFloat(curr.confidence), 0) / data.total_rows).toFixed(1);

                    document.getElementById('statTotal').innerText = data.total_rows;
                    document.getElementById('statMalignant').innerText = mCount;
                    document.getElementById('statBenign').innerText = bCount;
                    document.getElementById('statConfidence').innerText = avgConf + "%";
                    document.getElementById('tableBadge').innerText = `BATCH_${data.total_rows}`;

                    const note = document.getElementById('medicalNote');
                    const mPerc = Math.round(mCount/data.total_rows*100);
                    const bPerc = 100 - mPerc;

                    if (mCount > bCount) {
                        note.innerHTML = `<span class="text-red-400 font-black uppercase">Waspada:</span> Ditemukan dominansi kasus <span class="text-white">${mPerc}% Ganas</span>. Protokol klinis mendesak diperlukan untuk validasi histopatologi pada pasien terkait.`;
                    } else {
                        note.innerHTML = `<span class="text-emerald-400 font-black uppercase">Observasi:</span> Mayoritas data (<span class="text-white">${bPerc}% Jinak</span>) menunjukkan indikasi non-kanker. Tetap lakukan pemantauan rutin sesuai protokol skrining berkala.`;
                    }

                    updateChart(mCount, bCount);
                    
                    const tbody = document.getElementById('csvTableBody');
                    tbody.innerHTML = data.predictions.map((p, idx) => {
                        const labelIndo = p.prediction === 'Malignant' ? 'GANAS' : 'JINAK';
                        const colorClass = p.prediction === 'Malignant' ? 'text-red-400' : 'text-emerald-400';
                        return `
                            <tr class="hover:bg-slate-800/30 transition-all">
                                <td class="p-4 text-slate-500 font-bold uppercase tracking-tighter">PX-${String(idx + 1).padStart(3, '0')}</td>
                                <td class="p-4 font-bold ${colorClass}">${labelIndo}</td>
                                <td class="p-4 text-right text-slate-300">${p.confidence}</td>
                            </tr>
                        `;
                    }).join('');
                    setLoader(false);
                }, 2500);
            } catch (e) { 
                setLoader(false);
                alert("Gagal memproses batch CSV!"); 
            }
        }

        function showResult(type) {
            document.getElementById('resultArea').classList.remove('opacity-0');
            document.getElementById('resultArea').classList.add('opacity-100');
            document.getElementById('jsonDisplay').classList.toggle('hidden', type !== 'json');
            document.getElementById('csvDisplay').classList.toggle('hidden', type !== 'csv');
        }

        function updateChart(m, b) {
            const ctx = document.getElementById('predictionChart').getContext('2d');
            if (myChart) myChart.destroy();
            
            myChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['GANAS (Risiko Tinggi)', 'JINAK (Risiko Rendah)'],
                    datasets: [{
                        data: [m, b],
                        backgroundColor: [
                            'rgba(239, 68, 68, 0.9)', 
                            'rgba(16, 185, 129, 0.9)'
                        ],
                        hoverBackgroundColor: ['#ef4444', '#10b981'],
                        borderColor: '#020617',
                        borderWidth: 10,
                        hoverOffset: 20,
                        borderRadius: 20
                    }]
                },
                options: { 
                    cutout: '70%', 
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { 
                            position: 'bottom', 
                            labels: { 
                                color: '#94a3b8', 
                                font: { family: 'JetBrains Mono', size: 10, weight: 'bold' }, 
                                padding: 20,
                                usePointStyle: true
                            } 
                        },
                        tooltip: {
                            backgroundColor: '#020617',
                            padding: 12,
                            cornerRadius: 12,
                            borderWidth: 1,
                            borderColor: '#334155'
                        }
                    } 
                }
            });
        }

        const dataSet = { 
            malignant: { "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399, "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587, "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0, "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119, "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189 }, 
            benign: { "mean_radius": 13.54, "mean_texture": 14.36, "mean_perimeter": 87.46, "mean_area": 566.3, "mean_smoothness": 0.09779, "mean_compactness": 0.08129, "mean_concavity": 0.06664, "mean_concave_points": 0.04781, "mean_symmetry": 0.1885, "mean_fractal_dimension": 0.05766, "radius_error": 0.2699, "texture_error": 0.7886, "perimeter_error": 2.058, "area_error": 23.56, "smoothness_error": 0.008462, "compactness_error": 0.0146, "concavity_error": 0.02387, "concave_points_error": 0.01315, "symmetry_error": 0.0198, "fractal_dimension_error": 0.0023, "worst_radius": 15.11, "worst_texture": 19.26, "worst_perimeter": 99.7, "worst_area": 711.2, "worst_smoothness": 0.144, "worst_compactness": 0.1773, "worst_concavity": 0.239, "worst_concave_points": 0.1288, "worst_symmetry": 0.2977, "worst_fractal_dimension": 0.07259 } 
        };
        function fillData(t) { document.getElementById('jsonInput').value = JSON.stringify(dataSet[t], null, 2); }