function updateTimeForElement(element) {
    if (element.hasAttribute('data-first-call-time')){
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
        if (elapsed > 60){
            update_time = 10000
        }
        if (elapsed > 360){
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

