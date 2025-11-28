function updateTimeForElement(element) {
    if (element.hasAttribute('data-first-call-time')) {
        let firstCallTime = element.getAttribute('data-first-call-time');
        if (!firstCallTime) {
            firstCallTime = new Date().getTime();
            element.setAttribute('data-first-call-time', firstCallTime);
        } else {
            firstCallTime = parseInt(firstCallTime);
        }

        let now = new Date().getTime();
        let elapsed = (now - firstCallTime) / 1000; // convert milliseconds to seconds

        let displayText = formatElapsedTime(elapsed);
        element.innerText = displayText;
        update_time = 3000
        if (elapsed > 60) {
            update_time = 10000
        }
        if (elapsed > 360) {
            update_time = 100000
        }

        // Add a random number of milliseconds to the timeout to avoid all elements updating at the same time
        setTimeout(() => updateTimeForElement(element), update_time + Math.floor(Math.random() * 1000)); // call updateTimeForElement again after 3 seconds for this specific element
    }
}

function formatElapsedTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)} seconds` + " since loaded...";
    } else if (seconds < 3600) {
        return `${Math.round(seconds / 60)} minutes` + " since loaded...";
    } else if (seconds < 86400) {
        return `${Math.round(seconds / 3600)} hours` + " since loaded...";
    } else {
        return `${Math.round(seconds / 86400)} days` + " since loaded...";
    }
}

// Progress Wheel Functions
function updateProgress(current, total, description) {
    const container = document.getElementById('progress-wheel-container');
    const numerator = document.getElementById('progress-numerator');
    const denominator = document.getElementById('progress-denominator');
    const descriptionEl = document.querySelector('.progress-description');
    const circle = document.querySelector('.progress-circle-fill');
    const typingIndicator = document.getElementById('typing-indicator');

    if (!container || !numerator || !denominator || !circle) return;

    // Show the progress wheel and hide typing indicator
    container.style.display = 'block';
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }

    // Update text
    numerator.textContent = current;
    denominator.textContent = total;

    // Update description if provided
    if (description && descriptionEl) {
        descriptionEl.textContent = description;
    }

    // Calculate progress percentage
    const percentage = total > 0 ? (current / total) : 0;
    const circumference = 2 * Math.PI * 24; // r=24 for the smaller circle
    const offset = circumference - (percentage * circumference);

    // Update circle progress
    circle.style.strokeDashoffset = offset;

    // Hide when complete
    if (current >= total && total > 0) {
        setTimeout(() => {
            container.style.opacity = '0';
            container.style.transition = 'opacity 0.5s ease-out';
            setTimeout(() => {
                container.style.display = 'none';
                container.style.opacity = '1';
                container.style.transition = '';
                // Show typing indicator again if it should be visible
                if (typingIndicator) {
                    typingIndicator.style.display = 'flex';
                }
            }, 500);
        }, 1000);
    }
}

function hideProgress() {
    const container = document.getElementById('progress-wheel-container');
    const typingIndicator = document.getElementById('typing-indicator');
    if (container) {
        container.style.display = 'none';
    }
    // Restore typing indicator
    if (typingIndicator) {
        typingIndicator.style.display = 'flex';
    }
}

// Listen for WebSocket messages from htmx for progress updates
document.addEventListener('DOMContentLoaded', function () {
    document.body.addEventListener('htmx:wsAfterMessage', function (event) {
        try {
            const data = JSON.parse(event.detail.message);
            if (data.current !== undefined && data.total !== undefined) {
                updateProgress(data.current, data.total, data.description || '');
            }
        } catch (e) {
            // Not a progress message, ignore
        }
    });
});
