/**
 * stats.js
 * Handle statistics tracking (FPS, counts)
 */

import state from './state.js';

// DOM elements
let personCountElement, faceCountElement, fpsCounterElement;

/**
 * Initialize stats module
 * @param {Object} elements - DOM elements
 */
export function initStats(elements) {
    personCountElement = elements.personCountElement;
    faceCountElement = elements.faceCountElement;
    fpsCounterElement = elements.fpsCounterElement;
}

/**
 * Setup FPS counter
 */
export function setupFpsCounter() {
    // Clear existing timer if any
    if (state.fpsTimerId) {
        clearInterval(state.fpsTimerId);
    }
    
    // Setup new timer
    state.frameCount = 0;
    state.lastFrameTime = performance.now();
    
    state.fpsTimerId = setInterval(() => {
        const currentTime = performance.now();
        const elapsedTime = (currentTime - state.lastFrameTime) / 1000;
        
        if (elapsedTime > 0) {
            const fps = Math.round(state.frameCount / elapsedTime);
            fpsCounterElement.textContent = fps;
            
            // Reset counters
            state.frameCount = 0;
            state.lastFrameTime = currentTime;
        }
    }, 1000);
    
    console.log("FPS counter started");
}

/**
 * Update statistics display
 * @param {number} personCount - Number of detected persons
 * @param {number} faceCount - Number of detected faces
 */
export function updateStats(personCount, faceCount) {
    personCountElement.textContent = personCount;
    faceCountElement.textContent = faceCount;
}

/**
 * Reset statistics counters
 */
export function resetStats() {
    personCountElement.textContent = '0';
    faceCountElement.textContent = '0';
    fpsCounterElement.textContent = '0';
}