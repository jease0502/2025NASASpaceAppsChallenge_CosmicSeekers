// Centralized DOM element selectors
export const DOMElements = {
    // Persistent elements
    navLinks: document.getElementById('nav-links'),
    sceneContainer: document.getElementById('scene-container'),
    pageContainer: document.getElementById('page-container'),
    mainContent: document.getElementById('main-content'),
    homeBtn: document.getElementById('home-btn'),
};

// Function to query elements available only on specific pages
export function queryPageDOMElements() {
    DOMElements.initialPrompt = document.getElementById('initial-prompt');
    DOMElements.predictionCard = document.getElementById('prediction-card');
    DOMElements.cardTitle = document.getElementById('card-title');
    DOMElements.predictionDetails = document.getElementById('prediction-details');
    DOMElements.predictionSummary = document.getElementById('prediction-summary');
    DOMElements.infoPanel = document.getElementById('info-panel');
    DOMElements.infoName = document.getElementById('info-name');
    DOMElements.infoDetails = document.getElementById('info-details');
    DOMElements.cameraControls = document.getElementById('camera-controls');
    DOMElements.camPanUp = document.getElementById('cam-pan-up');
    DOMElements.camPanDown = document.getElementById('cam-pan-down');
    DOMElements.camPanLeft = document.getElementById('cam-pan-left');
    DOMElements.camPanRight = document.getElementById('cam-pan-right');
    DOMElements.camZoomIn = document.getElementById('cam-zoom-in');
    DOMElements.camZoomOut = document.getElementById('cam-zoom-out');
    DOMElements.camReset = document.getElementById('cam-reset');
    DOMElements.dashboardContainer = document.getElementById('dashboard-container');
    DOMElements.dashboardTitleKic = document.getElementById('dashboard-title-kic');
    DOMElements.backToExplorerBtn = document.getElementById('back-to-explorer-btn');
    DOMElements.planetSelectorContainer = document.getElementById('planet-selector-container');
    
    // --- START: ADDED/CORRECTED SECTION ---
    DOMElements.dashboardModelSelector = document.getElementById('dashboard-model-selector'); // This line was missing
    // --- END: ADDED/CORRECTED SECTION ---

    DOMElements.dashboardPredictionLabel = document.getElementById('dashboard-prediction-label');
    DOMElements.dashboardConfidenceText = document.getElementById('dashboard-confidence-text');
    DOMElements.dashboardConfidenceBar = document.getElementById('dashboard-confidence-bar');
    DOMElements.dashboardControls = {
        planet_radius: document.getElementById('planet_radius'), period: document.getElementById('period'),
        equilibrium_temp: document.getElementById('equilibrium_temp'), duration: document.getElementById('duration'),
        insolation_flux: document.getElementById('insolation_flux'), depth: document.getElementById('depth'),
        snr: document.getElementById('snr'), transit_midpoint: document.getElementById('transit_midpoint'),
        stellar_temp: document.getElementById('stellar_temp'), stellar_radius: document.getElementById('stellar_radius'),
        stellar_logg: document.getElementById('stellar_logg'),
    };
    DOMElements.dashboardValues = {
        planet_radius_val: document.getElementById('planet_radius_val'), period_val: document.getElementById('period_val'),
        equilibrium_temp_val: document.getElementById('equilibrium_temp_val'), duration_val: document.getElementById('duration_val'),
        insolation_flux_val: document.getElementById('insolation_flux_val'), depth_val: document.getElementById('depth_val'),
        snr_val: document.getElementById('snr_val'), transit_midpoint_val: document.getElementById('transit_midpoint_val'),
        stellar_temp_val: document.getElementById('stellar_temp_val'), stellar_radius_val: document.getElementById('stellar_radius_val'),
        stellar_logg_val: document.getElementById('stellar_logg_val'),
    };
    DOMElements.lightCurveContainer = document.getElementById('light-curve-container');
    DOMElements.lightCurveCanvas = document.getElementById('light-curve-canvas');
    DOMElements.speedControlContainer = document.getElementById('speed-control-container');
    DOMElements.speedSlider = document.getElementById('speed-slider');
    DOMElements.speedValue = document.getElementById('speed-value');
}