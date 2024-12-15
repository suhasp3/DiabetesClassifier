// Ensure the document is loaded before running JS
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const resultDiv = document.getElementById("result");

    // Prevent form submission and handle it with JavaScript
    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        // Collect form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Convert form data to JSON
        const jsonData = JSON.stringify(data);

        // Send POST request to the backend
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: jsonData,
        });

        // Parse and display the result
        const result = await response.json();
        resultDiv.innerHTML = `<h3>Prediction: ${result.prediction}</h3>`;
    });
});
