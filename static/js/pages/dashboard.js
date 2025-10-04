import { DOMElements, queryPageDOMElements } from '../dom.js';
import { ApiService } from '../api.js';
import { ThreeManager } from '../threeManager.js';
import { availableModels } from '../main.js';

let state = {
    params: {},
    predictionTimeout: null,
    baselinePlanetSystem: null,
    currentSystemData: null,
    eventController: null,
};

// ... (檔案中其餘的函式 updateVisuals, getPrediction 等保持不變)
function updateVisuals(key, value) { const systemGroup = ThreeManager.state.systemGroups[0]; if (!systemGroup) return; const starMesh = systemGroup.children[0]; if (!starMesh) return; const planetSystem = state.baselinePlanetSystem; switch(key) { case 'stellar_radius': starMesh.scale.setScalar(2 * Math.pow(value, 0.5)); break; case 'planet_radius': if (!planetSystem) break; const starScale = starMesh.scale.x; const starRadius = state.currentSystemData.star.stellar_radius; const planetRadiusInSolarRadii = (value / 109.2); const newPlanetScale = starScale * Math.pow(planetRadiusInSolarRadii, 0.5) / Math.pow(starRadius, 0.5) * 8; planetSystem.planetGroup.scale.setScalar(newPlanetScale); break; case 'period': if (!planetSystem) break; const baseOrbit = starMesh.scale.x * 4; const newRadius = baseOrbit + 2.5 * Math.pow(value, 0.6); planetSystem.orbitParams.radius = newRadius; const orbitLine = planetSystem.orbitGroup.children.find(c => c.type === 'Line'); if (orbitLine) { const points = []; for (let i = 0; i <= 128; i++) { const theta = (i / 128) * Math.PI * 2; points.push(new THREE.Vector3(Math.cos(theta) * newRadius, 0, Math.sin(theta) * newRadius)); } orbitLine.geometry.setFromPoints(points); orbitLine.geometry.computeBoundingSphere(); } break; case 'equilibrium_temp': if (!planetSystem) break; let activeTexture; if (value < 200) activeTexture = ThreeManager.state.assets.textures.ice; else if (value < 400) activeTexture = ThreeManager.state.assets.textures.earth; else if (value < 1000) activeTexture = ThreeManager.state.assets.textures.rocky; else activeTexture = ThreeManager.state.assets.textures.lava; planetSystem.planetMesh.material.map = activeTexture; planetSystem.planetMesh.material.needsUpdate = true; break; case 'stellar_temp': let starColor = 0xffffff; if (value < 4000) starColor = 0xff9966; else if (value < 5200) starColor = 0xffcc66; else if (value < 6000) starColor = 0xffffcc; else if (value < 7500) starColor = 0xffffff; else starColor = 0xccccff; starMesh.material.color.setHex(starColor); const pointLight = systemGroup.children.find(c => c.type === 'PointLight'); if(pointLight) pointLight.color.setHex(starColor); break; } }
async function getPrediction() { if (!DOMElements.dashboardModelSelector) return; const selectedModel = DOMElements.dashboardModelSelector.value; DOMElements.dashboardPredictionLabel.textContent = '...'; DOMElements.dashboardPredictionLabel.className = 'prediction-label'; const result = await ApiService.predictSystemData(selectedModel, [state.params]); if (result && !result.error) { const prediction = Array.isArray(result) ? result[0] : result; updateResult(prediction); } else { DOMElements.dashboardPredictionLabel.textContent = 'Error'; console.error("Prediction failed:", result?.error); } }
function updateResult(prediction) { const { prediction: label, confidence } = prediction; const confidencePercent = (confidence * 100).toFixed(0); DOMElements.dashboardPredictionLabel.textContent = label; DOMElements.dashboardPredictionLabel.className = `prediction-label ${label.replace(' ', '-')}`; DOMElements.dashboardConfidenceText.textContent = `Confidence: ${confidencePercent}%`; DOMElements.dashboardConfidenceBar.style.width = `${confidencePercent}%`; DOMElements.dashboardConfidenceBar.style.backgroundColor = label === 'CONFIRMED' ? '#34d399' : '#f87171'; }
function handleSliderChange(key, value) { const floatValue = parseFloat(value); if (DOMElements.dashboardValues[key + '_val']) { DOMElements.dashboardValues[key + '_val'].textContent = floatValue.toFixed(2); } state.params[key] = floatValue; updateVisuals(key, floatValue); clearTimeout(state.predictionTimeout); state.predictionTimeout = setTimeout(getPrediction, 250); }
function loadPlanetData(planetIndex) { const pIndex = parseInt(planetIndex, 10); DOMElements.planetSelectorContainer.querySelectorAll('.planet-selector-btn').forEach(btn => { btn.classList.toggle('active', btn.dataset.planetIndex == pIndex); }); const currentStar = state.currentSystemData.star; let baselinePlanet = state.currentSystemData.planets.find(p => p.planet_index === pIndex); state.baselinePlanetSystem = ThreeManager.state.planetSystems.find(ps => ps.data.planet_index === pIndex) || null; state.params = {}; if (baselinePlanet) { state.params.source_telescope = baselinePlanet.source_telescope; } else { state.params.source_telescope = currentStar.source_telescope; } for (const key in DOMElements.dashboardControls) { let value; if (key.startsWith('stellar_')) { value = currentStar?.[key] ?? 0; } else { value = baselinePlanet?.[key] ?? 0; } state.params[key] = parseFloat(value) || 0; if(DOMElements.dashboardControls[key]) { DOMElements.dashboardControls[key].value = value; if (DOMElements.dashboardValues[key + '_val']) { DOMElements.dashboardValues[key + '_val'].textContent = parseFloat(value).toFixed(2); } } updateVisuals(key, value); } getPrediction(); }
function setupInitialValues() { const { planets, star } = state.currentSystemData; DOMElements.planetSelectorContainer.innerHTML = ''; if (planets && planets.length > 0) { planets.forEach(planet => { const button = document.createElement('button'); const planetName = `${star.system_name}.${planet.planet_index.toString().padStart(2, '0')}`; button.className = 'planet-selector-btn'; button.dataset.planetIndex = planet.planet_index; button.textContent = planetName; DOMElements.planetSelectorContainer.appendChild(button); }); loadPlanetData(planets[0].planet_index); } else { loadPlanetData(-1); } }

export const Dashboard = {
    init(systemData) {
        queryPageDOMElements();
        state.currentSystemData = systemData;
        state.eventController = new AbortController();
        const { signal } = state.eventController;
        
        DOMElements.dashboardTitleKic.textContent = `Analyze: ${systemData.star.system_name}`;

        
        if (DOMElements.dashboardModelSelector) {
            let options = '';
            for (const [key, name] of Object.entries(availableModels)) {
                options += `<option value="${key}">${name}</option>`;
            }
            DOMElements.dashboardModelSelector.innerHTML = options;
            DOMElements.dashboardModelSelector.addEventListener('change', getPrediction, { signal });
        } else {
            console.error("Dashboard Model Selector not found in the DOM.");
        }

        ThreeManager.populateScene(systemData);
        ThreeManager.state.camera.position.set(0, 40, 60);
        ThreeManager.state.camera.lookAt(0, 0, 0);

        for (const key in DOMElements.dashboardControls) {
            if (DOMElements.dashboardControls[key]) {
                DOMElements.dashboardControls[key].addEventListener('input', (e) => handleSliderChange(key, e.target.value), { signal });
            }
        }
        
        const planetSelectorClickHandler = (e) => {
            if (e.target.matches('.planet-selector-btn')) {
                loadPlanetData(e.target.dataset.planetIndex);
            }
        };
        DOMElements.planetSelectorContainer.addEventListener('click', planetSelectorClickHandler, { signal });

        setupInitialValues();
    },
    update() { 
        if (ThreeManager.state.scene) {
            const systemGroup = ThreeManager.state.systemGroups[0];
            if(systemGroup) systemGroup.rotation.y += 0.0005;
        } 
    },
    cleanup() {
        console.log("Cleaning up Dashboard event listeners...");
        if (state.eventController) {
            state.eventController.abort();
            state.eventController = null;
        }
        clearTimeout(state.predictionTimeout);
    }
};