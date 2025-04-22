/**
 * config.js
 * Configuration settings for the application
 */

// Check if the device is mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Export application settings
const config = {
    // Detection settings
    showPersons: true,
    showFaces: true,
    showConfidence: true,
    personColor: '#e74c3c',
    faceColor: '#2ecc71',
    
    // Server settings
    serverUrl: '/process_frame',
    
    // Performance settings
    frameRate: isMobile ? 10 : 30, // Lower FPS on mobile devices
    
    // Device info
    isMobile: isMobile
};

export default config;