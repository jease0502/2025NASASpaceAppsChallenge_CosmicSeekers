import { DOMElements, queryPageDOMElements } from '../dom.js';
import { ThreeManager } from '../threeManager.js';
import { AppState } from '../main.js';

let state = {
    speedMultiplier: 10,
    brightnessHistory: [],
    ctx: null,
    planetsToTrack: [],
    overlay: null,
    eventController: null,
};

function resizeCanvasAndHistory(canvas) {
    if (!canvas) return false;
    const { clientWidth, clientHeight } = canvas;

    if (clientWidth === 0 || clientHeight === 0) return false;

    if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
        canvas.width = clientWidth;
        canvas.height = clientHeight;

        const historySize = Math.floor(canvas.width / 2);
        if (state.brightnessHistory.length !== historySize) {
            state.brightnessHistory = new Array(historySize).fill(1.0);
        }
    }
    return true;
}

function calculateBrightness() {
    if (state.planetsToTrack.length === 0) return 1.0;
    const systemGroup = ThreeManager.state.systemGroups[0];
    if (!systemGroup) return 1.0;
    const starMesh = systemGroup.children[0];
    if (!starMesh) return 1.0;

    const starRadius = starMesh.scale.x;
    let totalDip = 0;
    const visualExaggerationFactor = 80;

    for (const planetSystem of state.planetsToTrack) {
        const planetPos = new THREE.Vector3();
        planetSystem.planetGroup.getWorldPosition(planetPos);

        const isInFrontOfStar = planetPos.x > 0;
        const dist2D = Math.sqrt(planetPos.y * planetPos.y + planetPos.z * planetPos.z);

        if (isInFrontOfStar && dist2D < starRadius) {
            const transitFactor = 1.0 - (dist2D / starRadius);
            const planetRadius = planetSystem.planetGroup.scale.x;
            const dipRatio = (planetRadius * planetRadius) / (starRadius * starRadius);
            totalDip += dipRatio * visualExaggerationFactor * transitFactor;
        }
    }
    return Math.max(0.2, 1.0 - totalDip) + (Math.random() - 0.5) * 0.005;
}

function drawLightCurve() {
    if (!state.ctx) return;
    const canvas = DOMElements.lightCurveCanvas;
    resizeCanvasAndHistory(canvas);
    if (canvas.width === 0) return;

    const { width, height } = canvas;
    const { ctx, brightnessHistory } = state;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(10, 20, 30, 0.5)';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = height * (i / 4);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    ctx.beginPath();
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#34d399';
    ctx.shadowBlur = 5;

    const stepX = width / (brightnessHistory.length - 1);
    const yCenter = height / 2;

    brightnessHistory.forEach((value, index) => {
        const x = index * stepX;
        const y = yCenter - (value - 1.0) * (height * 0.4);
        index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
    ctx.setLineDash([5, 5]);
    ctx.moveTo(0, yCenter); ctx.lineTo(width, yCenter);
    ctx.stroke();
    ctx.setLineDash([]);
}

function applySpeedMultiplier() {
    ThreeManager.state.planetSystems.forEach(system => {
        if (!system.orbitParams.originalSpeed) {
            system.orbitParams.originalSpeed = system.orbitParams.speed;
        }
        system.orbitParams.speed = system.orbitParams.originalSpeed * state.speedMultiplier;
    });
}

export const Kepler = {
    init(systemData) {
        queryPageDOMElements();
        state.eventController = new AbortController();
        const { signal } = state.eventController;

        ThreeManager.populateScene(systemData);
        ThreeManager.state.camera.position.set(200, 0, 0);
        ThreeManager.state.camera.lookAt(0, 0, 0);

        ThreeManager.state.systemGroups.forEach(systemGroup => {
            systemGroup.rotation.set(0, 0, 0);
        });

        // 移除舊 overlay
        const existingOverlay = document.getElementById('telescope-overlay');
        if (existingOverlay) existingOverlay.remove();

        // 動態建立 overlay
        state.overlay = document.createElement('div');
        state.overlay.id = 'telescope-overlay';
        state.overlay.innerHTML = `<img src="/static/textures/telescope.png" 
            style="width: 100%; height: 100%; object-fit: cover;" alt="Telescope View">`;
        state.overlay.style.position = 'absolute';
        state.overlay.style.top = '0';
        state.overlay.style.left = '0';
        state.overlay.style.width = '100%';
        state.overlay.style.height = '100%';
        state.overlay.style.pointerEvents = 'none';
        state.overlay.style.zIndex = '10';

        DOMElements.sceneContainer.appendChild(state.overlay);

        // Light curve 初始化
        const canvas = DOMElements.lightCurveCanvas;
        if (canvas) {
            state.ctx = canvas.getContext('2d');
            resizeCanvasAndHistory(canvas);

            // resize listener
            window.addEventListener("resize", () => {
                resizeCanvasAndHistory(canvas);
            });
        }

        state.brightnessHistory = new Array(400).fill(1.0);
        const validPlanets = systemData.planets.filter(p => p.duration && p.depth && p.period);
        state.planetsToTrack = ThreeManager.state.planetSystems.filter(system =>
            validPlanets.some(pData => pData.planet_index === system.data.planet_index)
        );

        // 綁定速度控制
        if (DOMElements.speedSlider) {
            DOMElements.speedSlider.addEventListener('input', (e) => {
                state.speedMultiplier = parseInt(e.target.value, 10);
                DOMElements.speedValue.textContent = `${state.speedMultiplier}x`;
                applySpeedMultiplier();
            }, { signal });
        }

        applySpeedMultiplier();
    },

    update() {
        const newBrightness = calculateBrightness();
        state.brightnessHistory.push(newBrightness);
        state.brightnessHistory.shift();
        drawLightCurve();
    },

    cleanup() {
        console.log("Cleaning up Kepler view...");
        if (state.eventController) {
            state.eventController.abort();
            state.eventController = null;
        }
        if (state.overlay && DOMElements.sceneContainer.contains(state.overlay)) {
            DOMElements.sceneContainer.removeChild(state.overlay);
            state.overlay = null;
        }
        ThreeManager.state.planetSystems.forEach(system => {
            if (system.orbitParams.originalSpeed) {
                system.orbitParams.speed = system.orbitParams.originalSpeed;
                delete system.orbitParams.originalSpeed;
            }
        });
        state.ctx = null;
    }
};
