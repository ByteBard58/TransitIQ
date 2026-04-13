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

                if (data.error) {
                    throw new Error(data.error);
                }

                // Populate Results
                displayResults(data);
                openModal();

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during prediction. Please check your inputs.');
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
