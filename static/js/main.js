import { DOMElements, queryPageDOMElements } from './dom.js';
import { UIManager } from './uiManager.js';
import { ApiService } from './api.js';
import { ThreeManager } from './threeManager.js';
import { MainExplorer } from './pages/mainExplorer.js';
import { Dashboard } from './pages/dashboard.js';
import { Kepler } from './pages/kepler.js';
import { AudioManager } from './audioManager.js';
import { TrainingSystem } from './pages/trainingSystem.js';
import { Tutorial } from './pages/tutorial.js';

export const availableModels = { "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "lightgbm": "LightGBM", "logistic_regression": "Logistic Regression" };
export const AppState = { currentPage: null, systems: [], activeSystemIndex: 0, activePageLogic: null, canNavigate: true, allSystemsList: null, latestTrainedModelName: null, };
const PageLogic = { main: MainExplorer, dashboard: Dashboard, kepler: Kepler, training_system: TrainingSystem, tutorial: Tutorial };

function getNavConfig(currentSystem) {
    const navItems = [
        { type: 'link', label: 'Tutorial', icon: 'school', action: 'navigate', target: 'tutorial' },
    ];
    if (currentSystem) {
        navItems.push({ type: 'link', label: 'Dashboard', icon: 'query_stats', action: 'navigate', target: 'dashboard' });
        navItems.push({ type: 'link', label: 'Kepler View', icon: 'satellite_alt', action: 'navigate', target: 'kepler' });
    }
    navItems.push({ type: 'link', label: 'Model Training', icon: 'model_training', action: 'navigate', target: 'training_system' });
    navItems.push({ type: 'button', label: 'Model Evaluation', icon: 'analytics', action: 'showStats', id: 'model-stats-btn' });
    navItems.push({ type: 'search', label: 'Search System', icon: 'search', action: 'toggleSearch' });
    return navItems;
}

function getPlanetMusicTrack(planet) { if (!planet || planet.equilibrium_temp === null) return 'planet_rocky'; const temp = planet.equilibrium_temp; if (temp < 200) return 'planet_ice'; if (temp < 400) return 'planet_earthlike'; if (temp > 1000) return 'planet_lava'; return 'planet_rocky'; }

async function handleStateChange(action, payload) {
    if (!AppState.canNavigate && !['focus_system', 'explore_specific'].includes(action)) return;
    if (['navigate', 'explore_initial', 'return_to_main', 'explore_specific'].includes(action)) { AppState.canNavigate = false; }
    const pageContainer = DOMElements.pageContainer;
    const scrollablePages = ['tutorial', 'training_system'];
    let nextPageName = null;
    switch(action) {
        case 'navigate': nextPageName = payload.target; break;
        case 'explore_initial': case 'explore_specific': case 'return_to_main': nextPageName = 'main'; break;
    }
    const pagesWithoutMainScene = ['tutorial', 'training_system'];
    if (DOMElements.sceneContainer) {
        if (nextPageName && pagesWithoutMainScene.includes(nextPageName)) { DOMElements.sceneContainer.classList.remove('visible'); }
        else { DOMElements.sceneContainer.classList.add('visible'); }
    }
    if (nextPageName && scrollablePages.includes(nextPageName)) { pageContainer.classList.add('overflow-y-auto'); }
    else { pageContainer.classList.remove('overflow-y-auto'); }
    const isNavigation = ['navigate', 'explore_initial', 'return_to_main', 'explore_specific'].includes(action);
    if (AppState.activePageLogic?.cleanup && isNavigation) {
        AppState.activePageLogic.cleanup();
        AppState.activePageLogic = null; 
    }
    switch (action) {
        case 'navigate': const activeSystem = AppState.systems[AppState.activeSystemIndex]; if (payload.target === 'kepler') { AudioManager.playMusic('kepler'); } else if (payload.target === 'tutorial') { AudioManager.playMusic('start'); } else { const track = getPlanetMusicTrack(activeSystem?.planets?.[0]); AudioManager.playMusic(track); } if (AppState.currentPage === 'training_system') { AppState.latestTrainedModelName = null; } await UIManager.transitionToPage(payload.target, () => { AppState.currentPage = payload.target; AppState.activePageLogic = PageLogic[payload.target]; AppState.activePageLogic.init(payload.target === 'dashboard' || payload.target === 'kepler' ? activeSystem : null, handleStateChange); addNavListeners(); }); break;
        case 'explore_initial': UIManager.showLoadingPrompt(); const threeSystems = await ApiService.getTripleSystemData(); if(threeSystems && threeSystems.length >= 3) { AppState.systems = threeSystems; AppState.activeSystemIndex = 0; const currentSystem = AppState.systems[AppState.activeSystemIndex]; const track = getPlanetMusicTrack(currentSystem.planets?.[0]); AudioManager.playMusic(track); AppState.currentPage = 'main'; AppState.activePageLogic = PageLogic.main; AppState.activePageLogic.init(AppState.systems, handleStateChange); addNavListeners(); } break;
        case 'explore_specific': UIManager.showLoadingPrompt(); const newSystems = await ApiService.getSpecificTripleSystemData(payload.system_id); if(newSystems && newSystems.length >= 3) { AppState.systems = newSystems; AppState.activeSystemIndex = 0; const currentSystem = AppState.systems[AppState.activeSystemIndex]; const track = getPlanetMusicTrack(currentSystem.planets?.[0]); AudioManager.playMusic(track); AppState.currentPage = 'main'; AppState.activePageLogic = PageLogic.main; UIManager.showMainUI(); AppState.activePageLogic.init(AppState.systems, handleStateChange); addNavListeners(); } break;
        case 'focus_system': AppState.activeSystemIndex = payload.index; const newActiveSystem = AppState.systems[AppState.activeSystemIndex]; const newTrack = getPlanetMusicTrack(newActiveSystem.planets?.[0]); AudioManager.playMusic(newTrack); if (AppState.activePageLogic && AppState.activePageLogic.updateForNewSystem) { AppState.activePageLogic.updateForNewSystem(newActiveSystem); } addNavListeners(); break;
        case 'return_to_main': if (!AppState.systems || AppState.systems.length === 0) { handleStateChange('explore_initial'); return; } const returnTrack = getPlanetMusicTrack(AppState.systems[AppState.activeSystemIndex]?.planets?.[0]); AudioManager.playMusic(returnTrack); if (AppState.currentPage === 'training_system' || AppState.currentPage === 'tutorial') { AppState.latestTrainedModelName = null; } await UIManager.animatePageOut(() => { AppState.currentPage = 'main'; AppState.activePageLogic = PageLogic.main; AppState.activePageLogic.init(AppState.systems, handleStateChange); addNavListeners(); }); break;
        default: AudioManager.playMusic('start'); AppState.currentPage = 'main'; AppState.activePageLogic = PageLogic.main; AppState.activePageLogic.init(null, handleStateChange); addNavListeners();
    }
    setTimeout(() => { AppState.canNavigate = true; }, 800);
}

function addNavListeners() {
    const navConfig = getNavConfig(AppState.systems.length > 0 ? AppState.systems[AppState.activeSystemIndex] : null);
    UIManager.updateNavLinks(navConfig);
}

async function populateSystemSelector(prefix = '') {
    if (!AppState.allSystemsList) { AppState.allSystemsList = await ApiService.getAllSystems(); }
    const listItems = document.getElementById(`${prefix}system-list-items`);
    if (AppState.allSystemsList && listItems) {
        if(listItems.children.length > 0 && AppState.allSystemsList.length === listItems.children.length) return;
        listItems.innerHTML = '';
        AppState.allSystemsList.forEach(system => { const link = document.createElement('a'); link.href = '#'; link.dataset.systemId = system.system_id; link.textContent = system.name; listItems.appendChild(link); });
    }
}

function renderConfusionMatrix(confusionMatrixObject) {
    let labels = [];
    let matrix = [];
    if (Array.isArray(confusionMatrixObject)) {
        matrix = confusionMatrixObject;
    } else if (confusionMatrixObject && Array.isArray(confusionMatrixObject.matrix)) {
        matrix = confusionMatrixObject.matrix;
        labels = Array.isArray(confusionMatrixObject.labels) ? confusionMatrixObject.labels : [];
    } else {
        return '<p>Unable to display confusion matrix.</p>';
    }

    const n = matrix.length;
    if (!labels || labels.length !== n) {
        labels = Array.from({ length: n }, (_, i) => `Class ${i}`);
    }

    const headerCells = labels.map(label => `<th class="cm-label">${label}</th>`).join('');
    const rowsHtml = matrix.map((row, i) => {
        const cells = row.map(val => `<td class="cm-cell">${val}</td>`).join('');
        return `<tr><th class="cm-label">${labels[i]}</th>${cells}</tr>`;
    }).join('');

    return `
        <table class="confusion-matrix-table">
            <tr><td class="cm-empty"></td><th colspan="${n}" class="cm-header">Predicted</th></tr>
            <tr><th class="cm-header cm-header-actual"><span>Actual</span></th>${headerCells}</tr>
            ${rowsHtml}
        </table>
    `;
}

async function showModelStats() {
    const modal = document.getElementById('model-stats-modal');
    const container = document.getElementById('model-cards-container');
    if (!modal || !container) return;
    
    container.innerHTML = '<p class="text-gray-400">Loading model stats...</p>';
    modal.classList.add('visible');
    
    const stats = await ApiService.getModelStats();
    
    if (!stats) {
        container.innerHTML = '<p class="text-red-400">Could not load model data.</p>';
        return;
    }

    container.innerHTML = '';
    for (const modelKey in stats) {
        if (availableModels[modelKey]) {
            const modelData = stats[modelKey];
            const modelName = availableModels[modelKey];
            const accuracy = (modelData.classification_report.accuracy * 100).toFixed(2);
            
            // --- ★ 樣式修改 ★ ---
            // 為 model-card 加上與 training system 同樣風格的 class
            const cardHTML = `
                <div class="model-card bg-[#2E2E2E]/50 border border-white/20 p-5 rounded-xl">
                    <h3 class="text-xl font-bold text-white">${modelName}</h3>
                    <div class="text-2xl font-semibold text-green-400 mt-2">Accuracy: ${accuracy}%</div>
                    <h4 class="text-sm font-bold text-gray-400 mt-4">Confusion Matrix</h4>
                    ${renderConfusionMatrix(modelData.confusion_matrix)}
                </div>
            `;
            container.innerHTML += cardHTML;
        }
    }
}

function animate() { requestAnimationFrame(animate); let updateCallback = AppState.activePageLogic?.update ? AppState.activePageLogic.update.bind(AppState.activePageLogic) : null; ThreeManager.updateAnimation(updateCallback); }

async function main() {
    await ThreeManager.init();
    AudioManager.init();
    
    DOMElements.homeBtn.addEventListener('click', (e) => { e.preventDefault(); window.location.reload(); });
    DOMElements.pageContainer.addEventListener('click', (e) => {
        const backBtn = e.target.closest('#back-to-explorer-btn');
        if (backBtn) { e.preventDefault(); handleStateChange('return_to_main'); }
    });

    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileNav = document.getElementById('mobile-nav');
    if (mobileMenuBtn && mobileNav) {
        mobileMenuBtn.addEventListener('click', () => { mobileNav.classList.toggle('hidden'); });
    }
    
    const modal = document.getElementById('model-stats-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    if(modalCloseBtn) modalCloseBtn.addEventListener('click', () => modal.classList.remove('visible'));
    if(modal) modal.addEventListener('click', (e) => { if (e.target === modal) { modal.classList.remove('visible'); }});

    document.addEventListener('click', (e) => {
        const targetElement = e.target;
        const navItem = targetElement.closest('[data-action]');
        
        if (navItem) {
            e.preventDefault();
            const action = navItem.dataset.action;
            const target = navItem.dataset.target;
            switch (action) {
                case 'navigate':
                    handleStateChange('navigate', { target });
                    if (mobileNav && !mobileNav.classList.contains('hidden')) { mobileNav.classList.add('hidden'); }
                    break;
                case 'showStats':
                    showModelStats();
                    if (mobileNav && !mobileNav.classList.contains('hidden')) { mobileNav.classList.add('hidden'); }
                    break;
                case 'toggleSearch':
                    const isMobile = !!navItem.closest('#mobile-nav');
                    const prefix = isMobile ? 'mobile-' : '';
                    const systemList = document.getElementById(`${prefix}system-list`);
                    if (systemList) {
                        const otherPrefix = isMobile ? '' : 'mobile-';
                        const otherSystemList = document.getElementById(`${otherPrefix}system-list`);
                        if (otherSystemList) otherSystemList.classList.add('hidden');
                        systemList.classList.toggle('hidden');
                        if (!systemList.classList.contains('hidden')) { populateSystemSelector(prefix); }
                    }
                    break;
            }
            return;
        }
        const systemLink = targetElement.closest('a[data-system-id]');
        if (systemLink) {
            e.preventDefault();
            handleStateChange('explore_specific', { system_id: systemLink.dataset.systemId });
            document.querySelectorAll('.system-list').forEach(list => list.classList.add('hidden'));
            if (mobileNav && !mobileNav.classList.contains('hidden')) { mobileNav.classList.add('hidden'); }
        }
        if (!targetElement.closest('#nav-container') && !targetElement.closest('#mobile-nav')) {
            document.querySelectorAll('.system-list').forEach(list => list.classList.add('hidden'));
        }
        if (!targetElement.closest('#nav-container') && !targetElement.closest('#mobile-menu-btn')) {
            if (mobileNav && !mobileNav.classList.contains('hidden')) { mobileNav.classList.add('hidden'); }
        }
    });

    document.addEventListener('input', (e) => {
        const targetId = e.target.id;
        if (targetId.endsWith('system-search-input')) {
            const prefix = targetId.startsWith('mobile-') ? 'mobile-' : '';
            const searchTerm = e.target.value.toLowerCase();
            document.querySelectorAll(`#${prefix}system-list-items a`).forEach(sys => {
                sys.style.display = sys.textContent.toLowerCase().includes(searchTerm) ? 'block' : 'none';
            });
        }
    });

    window.addEventListener('resize', ThreeManager.handleResize, false);
    addNavListeners();
    handleStateChange('initial_load');
    animate();
}

main();