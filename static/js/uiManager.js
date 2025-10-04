import { DOMElements } from './dom.js';

export const UIManager = {
    async transitionToPage(targetPage, onPageReady) {
        const { pageContainer } = DOMElements;
        const currentPageElements = Array.from(pageContainer.children);
        if (currentPageElements.length > 0) { currentPageElements.forEach(el => { el.style.transform = 'scale(0.95)'; el.style.opacity = '0'; }); }
        const response = await fetch(`/partials/${targetPage}`);
        if (!response.ok) {
            console.error(`Failed to fetch partial: ${targetPage}.`);
            if (currentPageElements.length > 0) { currentPageElements.forEach(el => { el.style.transform = 'scale(1)'; el.style.opacity = '1'; }); }
            return;
        }
        const newPageHTML = await response.text();
        setTimeout(() => {
            pageContainer.innerHTML = '';
            const newPage = document.createElement('div');
            newPage.innerHTML = newPageHTML;
            const newPageContent = newPage.children[0];
            if (newPageContent) {
                newPageContent.style.transform = 'scale(1.05)';
                newPageContent.style.opacity = '0';
                pageContainer.appendChild(newPageContent);
                void newPageContent.offsetWidth;
                newPageContent.style.transform = 'scale(1)';
                newPageContent.style.opacity = '1';
                setTimeout(() => { if (onPageReady) onPageReady(); }, 500);
            } else if (onPageReady) {
                onPageReady();
            }
        }, 250);
    },
    animatePageOut(onComplete) {
        const { pageContainer } = DOMElements;
        const currentPage = pageContainer.children[0];
        if (currentPage) {
            currentPage.style.transform = 'scale(0.95)';
            currentPage.style.opacity = '0';
            setTimeout(() => { pageContainer.innerHTML = ''; if (onComplete) onComplete(); }, 500);
        } else {
            if (onComplete) onComplete();
        }
    },
    updateNavLinks(navConfig) {
        const navLinksContainer = document.getElementById('nav-links');
        const mobileNavContainer = document.getElementById('mobile-nav');
        if (!navLinksContainer || !mobileNavContainer) return;

        const createNavItemHTML = (item, isMobile = false) => {
            const id = isMobile && item.id ? `mobile-${item.id}` : item.id;
            const action = `data-action="${item.action}"`;
            const target = item.target ? `data-target="${item.target}"` : '';

            const content = `<span class="material-symbols-outlined">${item.icon}</span><span>${item.label}</span>`;

            switch (item.type) {
                case 'link':
                    return `<a href="#" class="nav-item" ${action} ${target}>${content}</a>`;
                case 'button':
                    return `<button id="${id || ''}" class="nav-item" ${action}>${content}</button>`;
                case 'search':
                    const prefix = isMobile ? 'mobile-' : '';
                    return `
                        <div id="${prefix}system-selector-container" class="nav-item-search">
                            <button id="${prefix}select-system-btn" class="nav-item" data-action="toggleSearch">
                                ${content}
                            </button>
                            <div id="${prefix}system-list" class="system-list hidden">
                                <input type="text" id="${prefix}system-search-input" placeholder="Search by name...">
                                <div id="${prefix}system-list-items"></div>
                            </div>
                        </div>`;
                default:
                    return '';
            }
        };
        
        navLinksContainer.innerHTML = navConfig.map(item => createNavItemHTML(item, false)).join('');
        mobileNavContainer.innerHTML = navConfig.map(item => createNavItemHTML(item, true)).join('');
    },
    showInfoPanel(name, detailsHTML) { if (!DOMElements.infoPanel) return; DOMElements.infoName.textContent = name; DOMElements.infoDetails.innerHTML = detailsHTML; DOMElements.infoPanel.classList.add('visible'); },
    hideInfoPanel() { if (DOMElements.infoPanel) DOMElements.infoPanel.classList.remove('visible'); },
    showInitialPrompt() { DOMElements.pageContainer.innerHTML = `<div class="initial-prompt" id="initial-prompt"><h2 class="text-3xl font-bold mb-4">Welcome to the Space</h2><p>Click to begin an exploration</p></div>`; },
    showLoadingPrompt() { DOMElements.pageContainer.innerHTML = `<div class="initial-prompt" id="initial-prompt"><h2 class="text-3xl font-bold mb-4 animate-pulse">Retrieving data from stardust...</h2></div>`; },
    showMainUI() {
         DOMElements.pageContainer.innerHTML = `
            <div id="main-explorer-wrapper">
                <div id="prediction-card" class="prediction-card ui-element"> <div class="flex flex-col gap-4"> <h2 class="text-lg font-bold text-white" id="card-title">System Analysis</h2> <div id="prediction-summary" class="text-xs space-y-1 p-3 bg-white/5 rounded-md"></div> <div class="grid grid-cols-3 gap-2 text-xs font-bold text-gray-400 border-b border-gray-600 pb-2"> <span>Object</span><span>AI Prediction Result</span><span class="text-right">Confidence</span> </div> <div class="text-xs space-y-2" id="prediction-details"></div> </div> </div>
                <div id="info-panel" class="bg-background-dark/70 backdrop-blur-lg rounded-xl p-4 shadow-2xl w-72 border border-primary/30 ui-element"> <h3 class="font-bold text-lg mb-2" id="info-name">Object Name</h3> <div class="text-xs space-y-1" id="info-details"></div> </div>
                <div id="main-view-prompt" class="absolute bottom-24 left-1/2 -translate-x-1/2 text-center text-gray-400 text-xs z-10 pointer-events-none"> <p>Click on a celestial body to focus. Use scroll to zoom, drag to rotate.</p> </div>
                <div id="camera-controls" class="ui-element"> <div id="camera-controls-drag-handle" class="drag-handle"></div> <div class="control-btn-grid"> <div class="d-pad"> <button id="cam-pan-up" class="camera-btn d-pad-up" title="Pan Up"><span class="material-symbols-outlined">arrow_drop_up</span></button> <button id="cam-pan-left" class="camera-btn d-pad-left" title="Pan Left"><span class="material-symbols-outlined">arrow_left</span></button> <button id="cam-reset" class="camera-btn d-pad-center" title="Reset View"><span class="material-symbols-outlined">my_location</span></button> <button id="cam-pan-right" class="camera-btn d-pad-right" title="Pan Right"><span class="material-symbols-outlined">arrow_right</span></button> <button id="cam-pan-down" class="camera-btn d-pad-down" title="Pan Down"><span class="material-symbols-outlined">arrow_drop_down</span></button> </div> <div class="zoom-controls"> <button id="cam-zoom-in" class="camera-btn" title="Zoom In"><span class="material-symbols-outlined">add</span></button> <button id="cam-zoom-out" class="camera-btn" title="Zoom Out"><span class="material-symbols-outlined">remove</span></button> </div> </div> <div class="music-controls"> <span id="current-track-name">N/A</span> <button id="mute-btn" title="Mute/Unmute"> <span id="mute-icon" class="material-symbols-outlined">volume_up</span> </button> </div> </div>
            </div>
         `;
    }
};