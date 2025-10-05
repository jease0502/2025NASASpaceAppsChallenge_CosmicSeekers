// mainExplorer.js
import { DOMElements, queryPageDOMElements } from '../dom.js';
import { UIManager } from '../uiManager.js';
import { ThreeManager } from '../threeManager.js';
import { ApiService } from '../api.js';
import { createInfoRows } from '../dataMapper.js';
import { AppState, availableModels } from '../main.js';
import { AudioManager } from '../audioManager.js';

const draggablePanelController = {
    isDragging: false, offsetX: 0, offsetY: 0, panel: null, boundDragMove: null, boundDragEnd: null,
    dragStart(event) {
        if (event.button !== 0) return;
        this.isDragging = true;
        this.panel = document.getElementById('camera-controls');
        if (!this.panel) return;
        this.panel.style.transition = 'none';
        this.offsetX = event.clientX - this.panel.offsetLeft;
        this.offsetY = event.clientY - this.panel.offsetTop;
        this.boundDragMove = this.dragMove.bind(this);
        this.boundDragEnd = this.dragEnd.bind(this);
        document.addEventListener('mousemove', this.boundDragMove);
        document.addEventListener('mouseup', this.boundDragEnd);
    },
    dragMove(event) {
        if (!this.isDragging) return;
        event.preventDefault();
        let newX = event.clientX - this.offsetX;
        let newY = event.clientY - this.offsetY;
        const bounds = DOMElements.pageContainer.getBoundingClientRect();
        newX = Math.max(0, Math.min(newX, bounds.width - this.panel.offsetWidth));
        newY = Math.max(0, Math.min(newY, bounds.height - this.panel.offsetHeight));
        this.panel.style.left = `${newX}px`;
        this.panel.style.top = `${newY}px`;
        this.panel.style.right = 'auto';
        this.panel.style.bottom = 'auto';
    },
    dragEnd() {
        if (!this.isDragging) return;
        this.isDragging = false;
        if (this.panel) { this.panel.style.transition = ''; }
        document.removeEventListener('mousemove', this.boundDragMove);
        document.removeEventListener('mouseup', this.boundDragEnd);
    },
    eventController: null,
    setup() {
        this.eventController = new AbortController();
        const handle = document.getElementById('camera-controls-drag-handle');
        if (handle) {
            handle.addEventListener('mousedown', this.dragStart.bind(this), { signal: this.eventController.signal });
        }
    },
    teardown() {
        if (this.isDragging) { this.dragEnd(); }
        if (this.eventController) { this.eventController.abort(); this.eventController = null; }
    }
};

const cameraController = { state: { isAnimatingCamera: false, cameraFocusTarget: null, cameraOffset: new THREE.Vector3(), animationTarget: { position: new THREE.Vector3(), lookAt: new THREE.Vector3() }, }, initialize(systemsData, onStateChange) { this.systemsData = systemsData; this.onStateChange = onStateChange; this.state.isAnimatingCamera = false; this.state.cameraFocusTarget = null; }, setAnimationTarget(position, lookAt, focusTarget = null) { this.state.isAnimatingCamera = true; this.state.animationTarget.position.copy(position); this.state.animationTarget.lookAt.copy(lookAt); this.state.cameraFocusTarget = focusTarget; }, updatePosition() { if (!ThreeManager.state.camera) return; if (this.state.isAnimatingCamera) { ThreeManager.state.camera.position.lerp(this.state.animationTarget.position, 0.05); ThreeManager.state.camera.lookAt(this.state.animationTarget.lookAt); if (ThreeManager.state.camera.position.distanceTo(this.state.animationTarget.position) < 0.1) { this.state.isAnimatingCamera = false; } } else if (this.state.cameraFocusTarget) { const targetWorldPosition = new THREE.Vector3(); this.state.cameraFocusTarget.getWorldPosition(targetWorldPosition); ThreeManager.state.camera.position.copy(targetWorldPosition).add(this.state.cameraOffset); ThreeManager.state.camera.lookAt(targetWorldPosition); } }, unfocus() { if (this.state.cameraFocusTarget) { this.state.cameraFocusTarget = null; UIManager.hideInfoPanel(); if (DOMElements.resetViewBtn) DOMElements.resetViewBtn.classList.remove('hidden'); } }, pan(x, y) { this.unfocus(); const { camera } = ThreeManager.state; if (!camera) return; const panSpeed = 0.5; camera.translateX(x * panSpeed); camera.translateY(y * panSpeed); }, dolly(delta) { this.unfocus(); const { camera } = ThreeManager.state; if (!camera) return; const dollySpeed = 2.0; const direction = new THREE.Vector3(); camera.getWorldDirection(direction); const prospectivePosition = camera.position.clone().addScaledVector(direction, delta * dollySpeed); if (prospectivePosition.length() > 3 && prospectivePosition.length() < 5000) { camera.position.copy(prospectivePosition); } }, resetView() { if (!this.systemsData || this.systemsData.length === 0) { console.warn("resetView called without system data. Aborting."); return; } const activeSystemData = this.systemsData[AppState.activeSystemIndex]; if (!activeSystemData) return; const starRadius = activeSystemData.star.stellar_radius > 0 ? activeSystemData.star.stellar_radius : 1.0; const starScale = 2 * Math.pow(starRadius, 0.5); const orbits = activeSystemData.planets.map(p => (starScale * 4) + 2.5 * Math.pow(p.period, 0.6)); const maxRadius = orbits.length > 0 ? Math.max(...orbits) : 20; const activeSystemGroup = ThreeManager.state.systemGroups.find(g => g.name === `SystemGroup_${AppState.activeSystemIndex}`); const lookAtTarget = activeSystemGroup ? activeSystemGroup.position.clone() : new THREE.Vector3(0,0,0); const position = new THREE.Vector3(0, maxRadius * 0.3, maxRadius * 1.3).add(lookAtTarget); this.setAnimationTarget(position, lookAtTarget, null); if(DOMElements.resetViewBtn) DOMElements.resetViewBtn.classList.add('hidden'); UIManager.hideInfoPanel(); }, };
function updatePredictionSummary() {
    if (!DOMElements.predictionDetails || !DOMElements.predictionSummary) return;
    const predictions = DOMElements.predictionDetails.children;
    let confirmedCount = 0, candidateCount = 0, fpCount = 0;
    for (const pred of predictions) {
        if (pred.querySelector('.text-green-400')) confirmedCount++;
        else if (pred.querySelector('.text-orange-400')) candidateCount++;
        else if (pred.querySelector('.text-red-400')) fpCount++;
    }
    const total = confirmedCount + candidateCount + fpCount;
    if (total === 0) {
        DOMElements.predictionSummary.innerHTML = '<p>Click on a planet to run AI prediction.</p>';
        return;
    }
    DOMElements.predictionSummary.innerHTML = `
        <p class="flex justify-between"><strong>Analyzed Objects:</strong> <span>${total}</span></p>
        <p class="flex justify-between"><strong>Confirmed Planets:</strong> <span class="text-green-400">${confirmedCount}</span></p>
        <p class="flex justify-between"><strong>Candidate:</strong> <span class="text-orange-400">${candidateCount}</span></p>
        <p class="flex justify-between"><strong>False Positives:</strong> <span class="text-red-400">${fpCount}</span></p>
    `;
}

async function addPredictionToCard(planetData, systemData) {
    if (!systemData) return;
    const planetName = `${systemData.star.system_name}.${planetData.planet_index.toString().padStart(2,'0')}`;
    if (document.getElementById(`pred-${planetName}`) || !DOMElements.predictionDetails) return;
    let predColor = 'text-red-400';
    if (planetData.model_prediction === 'CONFIRMED') predColor = 'text-green-400';
    else if (planetData.model_prediction === 'CANDIDATE') predColor = 'text-orange-400';
    const predictionEntry = document.createElement('div');
    predictionEntry.id = `pred-${planetName}`;
    predictionEntry.className = 'grid grid-cols-3 gap-2 items-center';
    predictionEntry.innerHTML = `<span>${planetName}</span><span class="${predColor} font-bold">${planetData.model_prediction}</span><span class="text-right">${planetData.model_confidence}%</span>`;
    DOMElements.predictionDetails.appendChild(predictionEntry);
    updatePredictionSummary();
    if (DOMElements.predictionCard) DOMElements.predictionCard.classList.add('visible');
}

function createModelSelectorHTML() {
    let options = '';
    const defaultModel = AppState.latestTrainedModelName || Object.keys(availableModels)[0];
    for (const [key, name] of Object.entries(availableModels)) {
        const isLatestTrained = AppState.latestTrainedModelName === key;
        const displayName = isLatestTrained ? `${name} (Your Trained Model)` : name;
        const selected = key === defaultModel ? 'selected' : '';
        options += `<option value="${key}" ${selected}>${displayName}</option>`;
    }
    
    // Show status indicator when user has trained model
    const statusIndicator = AppState.latestTrainedModelName ? 
        `<div class="text-xs text-green-400 mb-2">✓ Using your trained ${availableModels[AppState.latestTrainedModelName]} model</div>` : '';
    
    return `
        <div class="flex flex-col gap-2 mt-3">
            ${statusIndicator}
            <select id="info-panel-model-selector" class="bg-gray-700 border border-gray-600 rounded-md text-xs p-2 w-full">${options}</select>
            <button id="predict-btn" class="predict-button w-full">
                <span class="material-symbols-outlined">psychology</span>
                <span>AI Predict</span>
            </button>
        </div>
    `;
}

export const MainExplorer = {
    init(systemsData, onStateChange) {
        if (!systemsData || systemsData.length === 0) {
            UIManager.showInitialPrompt();
            const prompt = document.getElementById('initial-prompt');
            if (prompt) {
                prompt.addEventListener('click', () => {
                    AudioManager.userInteracted();
                    onStateChange('explore_initial');
                }, { once: true });
            }
            return;
        }
        UIManager.showMainUI();
        queryPageDOMElements();

        // --- ★ 修正: 重新加入這行，讓控制台可見 ---
        if (DOMElements.cameraControls) {
            DOMElements.cameraControls.classList.add('visible');
        }
        
        cameraController.initialize(systemsData, onStateChange);
        ThreeManager.populateScene(systemsData, AppState.activeSystemIndex);
        cameraController.resetView();
        if (DOMElements.cardTitle) DOMElements.cardTitle.textContent = `AI Model Report: ${systemsData[AppState.activeSystemIndex].star.system_name}`;
        updatePredictionSummary();
        draggablePanelController.setup();
        
        const eventController = new AbortController();
        this.eventController = eventController;

        const muteBtn = document.getElementById('mute-btn');
        const updateMuteButton = () => {
            const muteIcon = document.getElementById('mute-icon');
            const trackNameEl = document.getElementById('current-track-name');
            if (!muteIcon || !trackNameEl) return;
            muteIcon.textContent = AudioManager.isMuted() ? 'volume_off' : 'volume_up';
            trackNameEl.textContent = AudioManager.getCurrentTrack();
        };

        if (muteBtn) {
            muteBtn.addEventListener('click', () => {
                AudioManager.toggleMute();
                updateMuteButton();
            }, { signal: eventController.signal });
            updateMuteButton();
        }
        
        let isDragging = false, hasDragged = false, prevMouse = {x: 0, y: 0};
        
        async function predictSinglePlanet(planetData, planetSystem) {
            const selector = document.getElementById('info-panel-model-selector');
            const button = document.getElementById('predict-btn');
            if (!selector || !button) return;
            const selectedModel = selector.value;
            button.textContent = 'Predicting...';
            button.disabled = true;
            selector.disabled = true;
            
            // Automatically use user's trained model if available and selected
            const useLatest = !!(AppState.latestTrainedModelName && selectedModel === AppState.latestTrainedModelName);
            if (useLatest) { console.log(`Using user's trained model: ${selectedModel}`); }
            
            const result = await ApiService.predictSystemData(selectedModel, [planetData], useLatest);
            const activeSystemData = systemsData[AppState.activeSystemIndex];
            if (result && !result.error) {
                const predictionResult = result[0];
                planetData.model_prediction = predictionResult.prediction;
                planetData.model_confidence = (predictionResult.confidence * 100).toFixed(0);
                addPredictionToCard(planetData, activeSystemData);
                if (planetData.model_prediction === 'FALSE POSITIVE') {
                    ThreeManager.explodePlanet(planetSystem.planetGroup);
                    setTimeout(() => UIManager.hideInfoPanel(), 500);
                }
                const resultDiv = document.createElement('div');
                let resultColor = 'text-red-400', bg = 'bg-red-500/20';
                if (planetData.model_prediction === 'CONFIRMED') { resultColor = 'text-green-400'; bg = 'bg-green-500/20'; }
                else if (planetData.model_prediction === 'CANDIDATE') { resultColor = 'text-orange-400'; bg = 'bg-amber-500/20'; }
                resultDiv.className = `font-bold mt-3 p-2 rounded-lg text-center ${bg}`;
                resultDiv.innerHTML = `AI Result: <span class="${resultColor}">${planetData.model_prediction}</span>`;
                const container = button.parentElement;
                if(container) { container.innerHTML = ''; container.appendChild(resultDiv); }
            } else {
                button.textContent = 'Prediction Failed';
                setTimeout(() => {
                    if (button && selector) {
                        button.innerHTML = `<span class="material-symbols-outlined">psychology</span><span>AI Predict</span>`;
                        button.disabled = false;
                        selector.disabled = false;
                    }
                }, 3000);
            }
        }
        
        function handleSceneClick(event) {
            if (cameraController.state.isAnimatingCamera || draggablePanelController.isDragging) return;
            if (event.target.closest('.ui-element')) { return; }
            const activeIndex = AppState.activeSystemIndex;
            const mouse = new THREE.Vector2((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(mouse, ThreeManager.state.camera);
            const intersects = raycaster.intersectObjects(ThreeManager.state.scene.children, true);
            if (intersects.length > 0) {
                let clickedObj = intersects[0].object;
                let parentGroup = null;
                let temp = clickedObj;
                while(temp.parent) { if (temp.name.startsWith('SystemGroup_')) { parentGroup = temp; break; } temp = temp.parent; }
                if (parentGroup) {
                    const clickedSystemIndex = parseInt(parentGroup.name.split('_')[1], 10);
                    if (clickedSystemIndex === activeIndex) {
                        let targetObject = clickedObj;
                        while (targetObject.parent && !targetObject.name.startsWith('Planet_') && !targetObject.name.startsWith('Star_')) { targetObject = targetObject.parent; }
                        if (targetObject.name.startsWith('Star_')) {
                            const activeSystemData = systemsData[activeIndex]; const starDetails = createInfoRows(activeSystemData.star, 'star'); UIManager.showInfoPanel(activeSystemData.star.system_name, starDetails); const position = new THREE.Vector3().copy(parentGroup.position).add(new THREE.Vector3(targetObject.scale.x * 1.5, targetObject.scale.x * 0.8, targetObject.scale.x * 2.0)); cameraController.state.cameraOffset.subVectors(position, parentGroup.position); cameraController.setAnimationTarget(position, parentGroup.position, targetObject);
                        } else if (targetObject.name.startsWith('Planet_')) {
                            const planetIndex = parseInt(targetObject.name.split('_')[1], 10); const planetSystem = ThreeManager.state.planetSystems.find(s => s.data.planet_index === planetIndex && s.data.host_star_id == systemsData[activeIndex].star.system_id ); if (planetSystem) { const activeSystemData = systemsData[activeIndex]; const planetName = `${activeSystemData.star.system_name}.${planetSystem.data.planet_index.toString().padStart(2,'0')}`; const planetDetails = createInfoRows(planetSystem.data, 'planet'); UIManager.showInfoPanel(planetName, planetDetails); const infoDetails = document.getElementById('info-details'); if (infoDetails && !planetSystem.data.hasOwnProperty('model_prediction')) { infoDetails.insertAdjacentHTML('beforeend', createModelSelectorHTML()); document.getElementById('predict-btn').onclick = () => predictSinglePlanet(planetSystem.data, planetSystem); } const planetWorldPos = new THREE.Vector3(); planetSystem.planetGroup.getWorldPosition(planetWorldPos); const direction = planetWorldPos.clone().sub(planetSystem.parentSystemGroup.position).normalize(); const distanceFactor = Math.max(planetSystem.planetGroup.scale.x * 10, 2); const cameraPosition = new THREE.Vector3().copy(planetWorldPos).add(direction.multiplyScalar(distanceFactor)); cameraController.state.cameraOffset.subVectors(cameraPosition, planetWorldPos); cameraController.setAnimationTarget(cameraPosition, planetWorldPos, planetSystem.planetGroup); }
                        }
                    } else {
                        if (DOMElements.predictionCard) DOMElements.predictionCard.classList.remove('visible');
                        const targetSystemId = systemsData[clickedSystemIndex].star.system_id;
                        onStateChange('explore_specific', { system_id: targetSystemId });
                    }
                }
            }
        }
        DOMElements.sceneContainer.onmousedown = (e) => { isDragging = true; hasDragged = false; prevMouse = {x: e.clientX, y: e.clientY}; };
        DOMElements.sceneContainer.onmousemove = (e) => { if (!isDragging || cameraController.state.isAnimatingCamera) return; const deltaX = e.clientX - prevMouse.x; const deltaY = e.clientY - prevMouse.y; prevMouse = {x: e.clientX, y: e.clientY}; if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) { hasDragged = true; if (cameraController.state.cameraFocusTarget) cameraController.unfocus(); } if (!cameraController.state.cameraFocusTarget) { const activeSystemGroup = ThreeManager.state.systemGroups.find(g => g.name === `SystemGroup_${AppState.activeSystemIndex}`); if(activeSystemGroup) { activeSystemGroup.rotation.y += deltaX * 0.005; activeSystemGroup.rotation.x += deltaY * 0.005; activeSystemGroup.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, activeSystemGroup.rotation.x)); } } };
        DOMElements.sceneContainer.onmouseup = (e) => { if (!hasDragged) handleSceneClick(e); isDragging = false; };
        DOMElements.sceneContainer.onwheel = (e) => { if (cameraController.state.isAnimatingCamera) return; if(cameraController.state.cameraFocusTarget) cameraController.unfocus(); const activeSystemGroup = ThreeManager.state.systemGroups.find(g => g.name === `SystemGroup_${AppState.activeSystemIndex}`); if(activeSystemGroup) { const distance = ThreeManager.state.camera.position.distanceTo(activeSystemGroup.position); const dollySpeed = Math.max(distance * 0.01, 1.0); ThreeManager.state.camera.translateZ(e.deltaY * -0.01 * dollySpeed); } };
        
        // --- ★ 修正: 將 holdable button 的清理機制也納入 AbortController ---
        const setupHoldableButton = (button, action, signal) => {
            if (!button) return;
            let intervalId = null;
            const startAction = (event) => { event.preventDefault(); if (intervalId) return; action(); intervalId = setInterval(action, 50); };
            const stopAction = () => { if (intervalId) { clearInterval(intervalId); intervalId = null; } };
            button.addEventListener('mousedown', startAction, { signal });
            button.addEventListener('mouseup', stopAction, { signal });
            button.addEventListener('mouseleave', stopAction, { signal });
            button.addEventListener('touchstart', startAction, { signal });
            button.addEventListener('touchend', stopAction, { signal });
            button.addEventListener('touchcancel', stopAction, { signal });
        };
        setupHoldableButton(DOMElements.camPanUp, () => cameraController.pan(0, 1), eventController.signal);
        setupHoldableButton(DOMElements.camPanDown, () => cameraController.pan(0, -1), eventController.signal);
        setupHoldableButton(DOMElements.camPanLeft, () => cameraController.pan(-1, 0), eventController.signal);
        setupHoldableButton(DOMElements.camPanRight, () => cameraController.pan(1, 0), eventController.signal);
        setupHoldableButton(DOMElements.camZoomIn, () => cameraController.dolly(-1), eventController.signal);
        setupHoldableButton(DOMElements.camZoomOut, () => cameraController.dolly(1), eventController.signal);
        if (DOMElements.camReset) { DOMElements.camReset.addEventListener('click', () => cameraController.resetView(), { signal: eventController.signal }); }
    },
    update() { 
        cameraController.updatePosition(); 
    },
    cleanup() {
        console.log("Cleaning up MainExplorer event listeners...");
        draggablePanelController.teardown();
        if (this.eventController) {
            this.eventController.abort();
        }
        if(DOMElements.sceneContainer) {
            DOMElements.sceneContainer.onmousedown = null;
            DOMElements.sceneContainer.onmousemove = null;
            DOMElements.sceneContainer.onmouseup = null;
            DOMElements.sceneContainer.onwheel = null;
        }
    },
    updateForNewSystem(newSystemData) {
        if (DOMElements.cardTitle) {
            DOMElements.cardTitle.textContent = `AI Model Report: ${newSystemData.star.system_name}`;
        }
        if (DOMElements.predictionDetails) {
            DOMElements.predictionDetails.innerHTML = '';
        }
        updatePredictionSummary();
        UIManager.hideInfoPanel();
    }
};