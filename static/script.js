document.getElementById('archivo').addEventListener('change', function() {
    const fileInput = this;
    const columnsDisplay = document.getElementById('columnsDisplay');
    
    if (fileInput.files.length === 0) {
        columnsDisplay.innerHTML = '';
        return;
    }

    const formData = new FormData();
    formData.append('archivo', fileInput.files[0]);

    fetch('/get_columns', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            columnsDisplay.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            columnsDisplay.innerHTML = `
                <h4>Columnas del CSV:</h4>
                <div class="columns-row">
                    ${data.columns.map(col => `<span class="column-item">${col}</span>`).join('')}
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        columnsDisplay.innerHTML = `<p class="error">Error al obtener columnas: ${error.message}</p>`;
    });
});

// Handle prediction form submission
document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    const formData = new FormData(this);
    const predictionResults = document.getElementById('predictionResults');
    const predictionOutput = document.getElementById('predictionOutput');

    fetch('/make_prediction', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            predictionOutput.textContent = `Error: ${data.error}`;
            predictionResults.style.display = 'block';
        } else {
            predictionOutput.textContent = JSON.stringify(data.predictions, null, 2); // Pretty print JSON
            predictionResults.style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        predictionOutput.textContent = `Error: ${error.message}`;
        predictionResults.style.display = 'block';
    });
});