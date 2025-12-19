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
function updateProgress(current, total, description, analystName) {
    const container = document.getElementById('progress-wheel-container');
    const numerator = document.getElementById('progress-numerator');
    const denominator = document.getElementById('progress-denominator');
    const labelEl = document.getElementById('progress-label');
    const descriptionEl = document.querySelector('.progress-description');
    const circle = document.querySelector('.progress-circle-fill');
    const typingIndicator = document.getElementById('typing-indicator');

    if (!container || !numerator || !denominator || !circle) {
        console.error('Progress wheel elements not found');
        return;
    }

    console.log(`Progress update: ${current}/${total} - ${description} - ${analystName}`);

    // Force show the progress wheel with explicit styling
    container.style.display = 'block';
    container.style.opacity = '1';
    container.style.visibility = 'visible';

    // Force hide typing indicator
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
        typingIndicator.style.visibility = 'hidden';
    }

    // Update text
    numerator.textContent = current;
    denominator.textContent = total;

    // Update label and description
    if (labelEl) {
        if (analystName) {
            labelEl.textContent = analystName + ': Data Extraction';
        } else {
            labelEl.textContent = 'Data Extraction';
        }
    }

    if (description && descriptionEl) {
        descriptionEl.textContent = description;
    }

    // Calculate progress with two-phase transformation:
    // Phase 1 (0-50%): Linear lag showing 70% of actual
    // Phase 2 (50-100%): Exponential catch-up to reach 100%
    const x = total > 0 ? (current / total) : 0;
    let percentage;
    if (x <= 0.5) {
        // Phase 1: Linear Lag (30% drop) - at 50% actual, shows 35%
        percentage = x * 0.7;
    } else {
        // Phase 2: Exponential Catch-up from 35% to 100%
        const segment = (x - 0.5) / 0.5;
        percentage = 0.35 + (0.65 * Math.pow(segment, 1.5));
    }
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
                container.style.visibility = 'hidden';
                container.style.opacity = '1';
                container.style.transition = '';
                // Only show typing indicator if chat input is hidden (pending state)
                const chatInput = document.getElementById('chat-input');
                if (typingIndicator && chatInput && chatInput.style.display === 'none') {
                    typingIndicator.style.display = 'flex';
                    typingIndicator.style.visibility = 'visible';
                }
            }, 500);
        }, 1000);
    }
}

function hideProgress() {
    const container = document.getElementById('progress-wheel-container');
    const typingIndicator = document.getElementById('typing-indicator');
    const chatInput = document.getElementById('chat-input');

    if (container) {
        container.style.display = 'none';
    }

    // Only show typing indicator if chat input is hidden (meaning we're in pending state)
    if (typingIndicator && chatInput && chatInput.style.display === 'none') {
        typingIndicator.style.display = 'flex';
    }
}

// Context Usage Functions
var globalCanCompress = true;  // Track whether compression is possible

function updateContext(percentage, canCompress) {
    const indicator = document.getElementById('context-indicator');
    const percentageEl = document.getElementById('context-percentage');

    // Update global can_compress state
    if (canCompress !== undefined) {
        globalCanCompress = canCompress;
    }

    if (!indicator || !percentageEl) {
        return;
    }

    percentageEl.textContent = percentage + '%';

    // Update color based on percentage
    if (percentage >= 80) {
        indicator.classList.add('context-critical');
        indicator.classList.remove('context-warning');
    } else if (percentage >= 60) {
        indicator.classList.add('context-warning');
        indicator.classList.remove('context-critical');
    } else {
        indicator.classList.remove('context-warning', 'context-critical');
    }

    // Show the indicator
    indicator.style.opacity = '1';

    // If we hit 100%, show the context limit banner and hide input
    if (percentage >= 100) {
        showContextLimitBanner(globalCanCompress);
    }
}

async function compressContext() {
    const btn = document.querySelector('.compress-context-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Compressing... Please wait';
    }

    try {
        const response = await fetch('/compress_context', { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            // Update button to show we're waiting
            if (btn) {
                btn.textContent = 'Processing... This may take a minute';
            }

            // Wait for compression to complete (poll for changes or just wait)
            // The compression needs time to summarize each message with LLM
            // We'll wait and then reload - the websocket will also update if we're patient
            setTimeout(() => {
                // Remove the context limit banner
                const banner = document.querySelector('.context-limit-banner');
                if (banner) {
                    banner.remove();
                }
                const dynamicBanner = document.getElementById('context-limit-banner-dynamic');
                if (dynamicBanner) {
                    dynamicBanner.remove();
                }

                // Show input elements again
                const chatInput = document.getElementById('chat-input');
                const sendButton = document.getElementById('send-message');
                if (chatInput) chatInput.style.display = 'flex';
                if (sendButton) sendButton.style.display = 'flex';

                // Force reload the chat
                location.reload();
            }, 10000);  // Wait 10 seconds for compression to complete
        } else {
            alert('Compression failed: ' + (result.error || 'Unknown error'));
            if (btn) {
                btn.disabled = false;
                btn.textContent = 'Compress Conversation';
            }
        }
    } catch (error) {
        alert('Compression failed: ' + error.message);
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Compress Conversation';
        }
    }
}

function showContextLimitBanner(canCompress) {
    // Check if banner already exists (either dynamic or server-rendered)
    if (document.getElementById('context-limit-banner-dynamic') ||
        document.querySelector('.context-limit-banner')) {
        return;
    }

    // Default to true if not specified
    if (canCompress === undefined) {
        canCompress = globalCanCompress;
    }

    // Hide chat input elements
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-message');
    const typingIndicator = document.getElementById('typing-indicator');
    const progressContainer = document.getElementById('progress-wheel-container');

    if (chatInput) chatInput.style.display = 'none';
    if (sendButton) sendButton.style.display = 'none';
    if (typingIndicator) typingIndicator.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'none';

    // Build compress section HTML only if compression is possible
    const compressSection = canCompress ? `
            <div class="context-compress-section">
                <hr style="margin: 1rem 0; border-color: #d97706;">
                <strong>🗜️ Want to try compressing the conversation?</strong>
                <p>This will summarize the first 50% of tool calls and responses to free up context space. <strong>Save a checkpoint first!</strong> This cannot be undone.</p>
                <button class="compress-context-btn" onclick="compressContext()">
                    Compress Conversation
                </button>
            </div>
    ` : '';

    // Create and inject the banner
    const banner = document.createElement('div');
    banner.id = 'context-limit-banner-dynamic';
    banner.className = 'context-limit-banner';
    banner.innerHTML = `
        <div class="context-limit-icon">⚠️</div>
        <div class="context-limit-text">
            <strong>Context Limit Reached</strong>
            <p>I'm sorry, but you have reached the context limit of this Science Discussion. The conversation has grown too large for me to continue processing.</p>
            <p>You can still:</p>
            <ul>
                <li>📊 Extract and download your data from the Analysts panel</li>
                <li>📁 Browse your Analysis database</li>
                <li>💾 Save a checkpoint of your project</li>
            </ul>
            <p>To continue discussing, please <strong>create a new project</strong> with your papers.</p>
            ${compressSection}
        </div>
    `;

    // Insert banner at the end of the chat messages
    const chatPanel = document.getElementById('chat-panel-messages');
    if (chatPanel) {
        chatPanel.appendChild(banner);
        // Scroll to show the banner
        const chat = document.getElementById('chat');
        if (chat) {
            chat.scrollTop = chat.scrollHeight;
        }
    }
}

// Listen for WebSocket messages from htmx for progress and context updates
document.addEventListener('DOMContentLoaded', function () {
    // Small delay to let WebSocket establish before checking for initial progress
    setTimeout(function () {
        const container = document.getElementById('progress-wheel-container');
        const typingIndicator = document.getElementById('typing-indicator');

        // If progress is showing, make sure typing indicator is hidden
        if (container && container.style.display === 'block' && typingIndicator) {
            typingIndicator.style.display = 'none';
        }
    }, 100);

    document.body.addEventListener('htmx:wsAfterMessage', function (event) {
        try {
            const data = JSON.parse(event.detail.message);
            // Handle progress updates
            if (data.current !== undefined && data.total !== undefined) {
                updateProgress(data.current, data.total, data.description || '', data.analyst_name);
            }
            // Handle context updates
            if (data.type === 'context' && data.percentage !== undefined) {
                updateContext(data.percentage, data.can_compress);
            }
            // Handle pause complete event - silently close modal and reset button
            if (data.type === 'pause_complete') {
                // Close the pause pending modal
                const pauseModal = document.getElementById('pause-pending-modal');
                if (pauseModal) {
                    pauseModal.style.display = 'none';
                }
                // Reset the pause button
                const pauseBtn = document.getElementById('pause-button');
                if (pauseBtn) {
                    pauseBtn.innerHTML = '<span data-text="Pause">⏸️</span>';
                    pauseBtn.disabled = false;
                }
            }
        } catch (e) {
            // Not a JSON message, ignore
        }
    });
});

// Global file viewer functions for pi_generated files

async function viewCSV(url) {
    // Create overlay div
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    overlay.style.zIndex = '999';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';

    // Create container for the grid
    const gridDiv = document.createElement('div');
    gridDiv.style.height = '80%';
    gridDiv.style.width = '90%';
    gridDiv.classList.add('ag-theme-alpine');

    // Append gridDiv to the overlay
    overlay.appendChild(gridDiv);

    // Create an exit button
    const exitButton = document.createElement('button');
    exitButton.textContent = '✕';
    exitButton.style.position = 'absolute';
    exitButton.style.top = '20px';
    exitButton.style.right = '20px';
    exitButton.style.zIndex = '1000';
    overlay.appendChild(exitButton);

    // Append overlay to body
    document.body.appendChild(overlay);

    // Fetch and parse CSV, then display in ag-Grid
    const response = await fetch(url);
    const csvText = await response.text();
    Papa.parse(csvText, {
        header: true,
        complete: function (results) {
            const columnDefs = Object.keys(results.data[0]).map(key => ({ headerName: key, field: key }));
            const gridOptions = {
                columnDefs: columnDefs,
                rowData: results.data,
                defaultColDef: {
                    resizable: true,
                    sortable: true,
                    filter: true
                }
            };

            // Create the grid
            const gridApi = agGrid.createGrid(gridDiv, gridOptions);
        }
    });

    // Event listener to remove the overlay
    exitButton.addEventListener('click', () => {
        document.body.removeChild(overlay);
    });
}

async function viewJSON(url) {
    // Create overlay
    const overlay = $('<div>').css({
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        zIndex: 999,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center'
    }).appendTo('body');

    // Create container for JSON viewer
    const viewer = $('<div>').css({
        width: '90%',
        height: '90%',
        backgroundColor: '#fff',
        overflow: 'auto',
        borderRadius: '8px',
        padding: '1rem'
    }).appendTo(overlay);

    // Fetch and display JSON
    try {
        const response = await fetch(url);
        const data = await response.json();
        viewer.jsonViewer(data, { collapsed: true, rootCollapsable: false });
    } catch (error) {
        console.error("Error processing JSON data: ", error);
        viewer.text("Error displaying JSON data.");
    }

    // Create refresh button
    $('<button>').html('&#x21bb;').css({
        position: 'absolute',
        top: '1vh',
        right: '5vw',
        fontSize: '1.5vw',
        padding: '0.5vw 1vw',
        zIndex: 1000
    }).click(async function () {
        viewer.empty();
        try {
            const response = await fetch(url);
            const data = await response.json();
            viewer.jsonViewer(data, { collapsed: true, rootCollapsable: false });
        } catch (error) {
            console.error("Error processing JSON data: ", error);
            viewer.text("Error displaying JSON data.");
        }
    }).appendTo(overlay);

    // Create exit button
    $('<button>').text('✕').css({
        position: 'absolute',
        top: '1vh',
        right: '1vw',
        fontSize: '1.5vw',
        padding: '0.5vw 1vw',
        zIndex: 1000
    }).click(function () {
        overlay.remove();
    }).appendTo(overlay);
}

async function viewPDF(url) {
    // Create overlay div
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    overlay.style.zIndex = '999';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';

    // Create iframe for the PDF
    const iframe = document.createElement('iframe');
    iframe.src = url;
    iframe.style.width = '90%';
    iframe.style.height = '90%';
    iframe.style.border = 'none';

    // Create an exit button
    const exitButton = document.createElement('button');
    exitButton.textContent = '✕';
    exitButton.style.position = 'absolute';
    exitButton.style.top = '20px';
    exitButton.style.right = '20px';
    exitButton.style.zIndex = '1000';

    // Append iframe and exit button to the overlay
    overlay.appendChild(iframe);
    overlay.appendChild(exitButton);

    // Append overlay to body
    document.body.appendChild(overlay);

    // Event listener to remove the overlay
    exitButton.addEventListener('click', () => {
        document.body.removeChild(overlay);
    });
}
