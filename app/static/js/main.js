document.addEventListener('DOMContentLoaded', () => {
    // Smooth Scroll
    const scrollBtn = document.getElementById('scroll-to-form');
    if (scrollBtn) {
        scrollBtn.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('form-section').scrollIntoView({ 
                behavior: 'smooth' 
            });
        });
    }

    // Input Animation & Label Handling
    const inputs = document.querySelectorAll('.input-field');
    inputs.forEach(input => {
        // Trigger label animation on load if value exists
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

        // Animate in
        setTimeout(() => notification.classList.add('active'), 10);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('active');
            setTimeout(() => notification.remove(), 400);
        }, 5000);
    }

    // Form Submission
    const form = document.getElementById('predictForm');
    const modal = document.getElementById('resultModal');
    const closeModalBtn = document.getElementById('closeModal');
    const resultsContainer = document.getElementById('resultsContainer');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = form.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            
            // Loading State
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            submitBtn.disabled = true;

            // Collect Data
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
                                // Map technical field names to human labels
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

                // Populate Results
                displayResults(data);
                openModal();
                showNotification('Analysis complete! Check the results.', 'success');

            } catch (error) {
                console.error('Error:', error);
                showNotification(error.message, 'error');
            } finally {
                // Reset Button
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

    // Close on click outside
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
        
        // Set Prediction Text
        predictionEl.textContent = data.prediction;
        
        // Clear previous bars
        barsContainer.innerHTML = '';

        // Sort probabilities
        const sortedProbs = Object.entries(data.probabilities)
            .sort(([,a], [,b]) => b - a);

        // Create Bars
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
            
            // Animate bar after a slight delay
            setTimeout(() => {
                item.querySelector('.prob-bar-fill').style.width = `${percentage}%`;
            }, 100);
        });
    }
});
