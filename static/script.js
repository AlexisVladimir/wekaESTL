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
                <table class="columns-table">
                    <thead>
                        <tr>
                            <th>Columna</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.columns.map(col => `<tr><td>${col}</td></tr>`).join('')}
                    </tbody>
                </table>
            `;
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        columnsDisplay.innerHTML = `<p class="error">Error al obtener columnas: ${error.message}</p>`;
    });
});