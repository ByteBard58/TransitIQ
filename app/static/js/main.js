document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const tabLinks = document.querySelectorAll('.tab-link:not([href])');
    const tabContents = document.querySelectorAll('.tab-content');

    tabLinks.forEach(link => {
        link.addEventListener('click', () => {
            const targetTab = link.dataset.tab;

            tabLinks.forEach(l => l.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            link.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });

    // Smooth Scroll to Form (from Welcome tab)
    const scrollBtn = document.getElementById('scroll-to-form');
    if (scrollBtn) {
        scrollBtn.addEventListener('click', (e) => {
            e.preventDefault();
            // Switch to Predict tab first
            tabLinks.forEach(l => l.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            document.querySelector('[data-tab="home"]').classList.add('active');
            document.getElementById('home-tab').classList.add('active');
            // Then scroll to form
            setTimeout(() => {
                document.getElementById('form-section').scrollIntoView({ 
                    behavior: 'smooth' 
                });
            }, 50);
        });
    }

    // Go to Batch Tab (from Welcome tab)
    const batchBtn = document.getElementById('go-to-batch');
    if (batchBtn) {
        batchBtn.addEventListener('click', (e) => {
            e.preventDefault();
            tabLinks.forEach(l => l.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            document.querySelector('[data-tab="batch"]').classList.add('active');
            document.getElementById('batch-tab').classList.add('active');
            setTimeout(() => {
                document.getElementById('batch-tab').scrollIntoView({ 
                    behavior: 'smooth' 
                });
            }, 50);
        });
    }

    // Input Animation & Label Handling
    const inputs = document.querySelectorAll('.input-field');
    inputs.forEach(input => {
        if (input.value) {
            input.classList.add('has-value');
        }
        
        input.addEventListener('input', () => {
            if (input.value.trim() !== '') {
                input.classList.add('has-value');
            } else {
                input.classList.remove('has-value');
            }
        });
    });

    // Field Labels for Human-Readable Errors
    const fieldLabels = {
        'koi_period': 'Orbital Period',
        'koi_time0bk': 'Transit Epoch',
        'koi_depth': 'Transit Depth',
        'koi_prad': 'Planet Radius',
        'koi_sma': 'Semi-Major Axis',
        'koi_incl': 'Inclination',
        'koi_teq': 'Equilibrium Temp',
        'koi_insol': 'Insolation Flux',
        'koi_impact': 'Impact Parameter',
        'koi_ror': 'Planet/Star Radius Ratio',
        'koi_srho': 'Stellar Density',
        'koi_dor': 'Planet-Star Distance',
        'koi_num_transits': 'Number of Transits'
    };

    // Notification System
    function showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: 'fa-circle-check',
            error: 'fa-circle-exclamation',
            info: 'fa-circle-info'
        };

        notification.innerHTML = `
            <i class="fa-solid ${icons[type]} notification-icon"></i>
            <span>${message}</span>
        `;

        container.appendChild(notification);

        setTimeout(() => notification.classList.add('active'), 10);

        setTimeout(() => {
            notification.classList.remove('active');
            setTimeout(() => notification.remove(), 400);
        }, 5000);
    }

    // Form Submission (Single Prediction)
    const form = document.getElementById('predictForm');
    const modal = document.getElementById('resultModal');
    const closeModalBtn = document.getElementById('closeModal');
    const resultsContainer = document.getElementById('resultsContainer');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = form.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            submitBtn.disabled = true;

            const formData = {};
            inputs.forEach(input => {
                formData[input.id] = parseFloat(input.value);
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                if (!response.ok) {
                    let errorMsg = 'An error occurred during prediction.';
                    if (data.detail) {
                        if (Array.isArray(data.detail)) {
                            errorMsg = data.detail.map(d => {
                                const field = d.loc[d.loc.length - 1];
                                const label = fieldLabels[field] || field;
                                return `${label}: ${d.msg}`;
                            }).join('\n');
                        } else {
                            errorMsg = data.detail;
                        }
                    }
                    throw new Error(errorMsg);
                }

                displayResults(data);
                openModal();
                showNotification('Analysis complete! Check the results.', 'success');

            } catch (error) {
                console.error('Error:', error);
                showNotification(error.message, 'error');
            } finally {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }

    // Modal Functions
    function openModal() {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', closeModal);
    }

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    }

    // Helper: Display Results
    function displayResults(data) {
        const predictionEl = document.getElementById('predictionResult');
        const barsContainer = document.getElementById('probabilityBars');
        
        predictionEl.textContent = data.prediction;
        
        barsContainer.innerHTML = '';

        const sortedProbs = Object.entries(data.probabilities)
            .sort(([,a], [,b]) => b - a);

        sortedProbs.forEach(([label, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            
            const item = document.createElement('div');
            item.className = 'prob-item';
            
            item.innerHTML = `
                <div class="prob-label">
                    <span>${label}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
            `;
            
            barsContainer.appendChild(item);
            
            setTimeout(() => {
                item.querySelector('.prob-bar-fill').style.width = `${percentage}%`;
            }, 100);
        });
    }

    // =========================================
    // Batch Prediction
    // =========================================
    
    // File input display
    const csvFileInput = document.getElementById('csvFile');
    const fileNameDisplay = document.getElementById('fileName');
    
    if (csvFileInput) {
        csvFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                fileNameDisplay.style.color = 'var(--accent-primary)';
            } else {
                fileNameDisplay.textContent = '';
            }
        });
    }

    // Batch Form Submission
    const batchForm = document.getElementById('batchForm');
    const batchModal = document.getElementById('batchResultModal');
    const closeBatchModalBtn = document.getElementById('closeBatchModal');

    if (batchForm) {
        batchForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = batchForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            submitBtn.disabled = true;

            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];

            if (!file) {
                showNotification('Please select a CSV file', 'error');
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict/batch', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    let errorMsg = 'An error occurred during batch prediction.';
                    if (data.detail) {
                        errorMsg = data.detail;
                    }
                    throw new Error(errorMsg);
                }

                displayBatchResults(data);
                openBatchModal();
                showNotification('Batch prediction complete!', 'success');

            } catch (error) {
                console.error('Error:', error);
                showNotification(error.message, 'error');
            } finally {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }

    function openBatchModal() {
        batchModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeBatchModal() {
        batchModal.classList.remove('active');
        document.body.style.overflow = '';
    }

    if (closeBatchModalBtn) {
        closeBatchModalBtn.addEventListener('click', closeBatchModal);
    }

    if (batchModal) {
        batchModal.addEventListener('click', (e) => {
            if (e.target === batchModal) {
                closeBatchModal();
            }
        });
    }

    function displayBatchResults(data) {
        const pieChartContainer = document.getElementById('pieChartContainer');
        const tableBody = document.getElementById('batchResultsTableBody');
        
        pieChartContainer.innerHTML = '';
        tableBody.innerHTML = '';

        // Transform API response to results array
        // Handle both "prediction_probability" and "predction_probability" (API typo)
        const probabilities = data.prediction_probability || data.predction_probability || [];
        
        if (!data.predicted_labels || !Array.isArray(data.predicted_labels) || probabilities.length === 0) {
            console.error('Invalid API response:', data);
            showNotification('Invalid response from server', 'error');
            return;
        }

        const results = data.predicted_labels.map((prediction, index) => {
            const probs = probabilities[index] || [];
            const maxProb = probs.length > 0 ? Math.max(...probs) : 0;
            return { prediction, confidence: maxProb };
        });

        // Count predictions by class
        const classCounts = {};
        results.forEach(result => {
            classCounts[result.prediction] = (classCounts[result.prediction] || 0) + 1;
        });

        const total = results.length;
        
        // Colors for each class
        const classColors = {
            'CONFIRMED': '#00ff88',
            'CANDIDATE': '#ffab00',
            'FALSE POSITIVE': '#ff4b2b'
        };

        // Calculate angles for pie chart
        let currentAngle = 0;
        const conicGradientParts = [];
        
        Object.entries(classCounts).forEach(([className, count]) => {
            const percentage = (count / total) * 100;
            const angle = (count / total) * 360;
            const startAngle = currentAngle;
            const endAngle = currentAngle + angle;
            const color = classColors[className] || '#888';
            
            conicGradientParts.push(`${color} ${startAngle}deg ${endAngle}deg`);
            currentAngle = endAngle;
        });

        // Create pie chart
        const pieChart = document.createElement('div');
        pieChart.className = 'pie-chart';
        pieChart.style.background = `conic-gradient(${conicGradientParts.join(', ')})`;
        pieChartContainer.appendChild(pieChart);

        // Create legend
        const legend = document.createElement('div');
        legend.className = 'pie-legend';
        
        Object.entries(classCounts).forEach(([className, count]) => {
            const percentage = ((count / total) * 100).toFixed(1);
            const color = classColors[className] || '#888';
            
            const legendItem = document.createElement('div');
            legendItem.className = 'pie-legend-item';
            legendItem.innerHTML = `
                <div class="pie-legend-color" style="background: ${color}"></div>
                <span>${className}: ${count} (${percentage}%)</span>
            `;
            legend.appendChild(legendItem);
        });
        
        pieChartContainer.appendChild(legend);

        // Populate table
        results.forEach((result, index) => {
            const row = document.createElement('tr');
            
            const confidenceClass = getConfidenceClass(result.confidence);
            const predictionClass = getPredictionClass(result.prediction);
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td class="prediction-cell ${predictionClass}">${result.prediction}</td>
                <td class="${confidenceClass}">${(result.confidence * 100).toFixed(1)}%</td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    function getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'high-confidence';
        if (confidence >= 0.5) return 'medium-confidence';
        return 'low-confidence';
    }

    function getPredictionClass(prediction) {
        if (prediction === 'CONFIRMED') return 'class-confirmed';
        if (prediction === 'CANDIDATE') return 'class-candidate';
        return 'class-false';
    }
});
