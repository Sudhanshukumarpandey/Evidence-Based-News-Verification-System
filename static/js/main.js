// Wait for the DOM to be fully loaded before attaching event listeners
document.addEventListener("DOMContentLoaded", function () {
    
    // 1. Loading State on Prediction Form Submission
    // Why: When predictions run, preprocessing and model execution can take a moment.
    // Showing a loading state improves User Experience (UX) and prevents multiple clicks.
    const predictForm = document.getElementById("predictForm");
    if (predictForm) {
        predictForm.addEventListener("submit", function () {
            const submitBtn = document.getElementById("submitBtn");
            if (submitBtn) {
                // Change button text and disable it to prevent multiple submissions
                submitBtn.innerHTML = "Processing Prediction...";
                submitBtn.disabled = true;
                submitBtn.style.opacity = "0.7";
                submitBtn.style.cursor = "not-allowed";
            }
        });
    }

    // 2. Confirmation before clearing all history
    // Why: Users might accidentally click "Clear History" and lose all their logs.
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", function (event) {
            // Confirm returns true if user clicks "OK", false if they click "Cancel"
            const confirmClear = confirm("Are you sure you want to delete ALL prediction history? This action cannot be undone.");
            if (!confirmClear) {
                // Prevent the browser from navigating to the clear-history link
                event.preventDefault();
            }
        });
    }

    // 3. Confirmation before deleting a single history entry
    // Why: Prevent accidental single deletions.
    const deleteButtons = document.querySelectorAll(".btn-delete-entry");
    deleteButtons.forEach(function (button) {
        button.addEventListener("click", function (event) {
            const confirmDelete = confirm("Are you sure you want to delete this prediction entry?");
            if (!confirmDelete) {
                event.preventDefault();
            }
        });
    });
});
